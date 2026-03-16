package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.runtime.FlashAttentionHelper
import io.aatricks.llmedge.LLMEdge
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

data class ImageGenerationRequest(
    val prompt: String,
    val negative: String = "",
    val width: Int = 512,
    val height: Int = 512,
    val steps: Int = 20,
    val cfgScale: Float = 7.0f,
    val seed: Long = -1L,
    val flashAttention: Boolean = true,
    val forceSequentialLoad: Boolean = false,
    val easyCache: EasyCacheParams = EasyCacheParams(),
    val loraModelDir: String? = null,
    val loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
    val model: ModelSpec? = null,
)

data class VideoGenerationRequest(
    val prompt: String,
    val negative: String = "",
    val width: Int = 512,
    val height: Int = 512,
    val videoFrames: Int = 16,
    val steps: Int = 20,
    val cfgScale: Float = 7.0f,
    val seed: Long = -1L,
    val flowShift: Float = Float.POSITIVE_INFINITY,
    val flashAttention: Boolean = true,
    val forceSequentialLoad: Boolean = false,
    val initImage: Bitmap? = null,
    val strength: Float = 1.0f,
    val sampleMethod: SampleMethod = SampleMethod.DEFAULT,
    val scheduler: Scheduler = Scheduler.DEFAULT,
    val easyCache: EasyCacheParams = EasyCacheParams(),
    val loraModelDir: String? = null,
    val loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
    val taehv: ModelSpec? = null,
    val model: ModelSpec? = null,
    val vae: ModelSpec? = null,
    val textEncoder: ModelSpec? = null,
) {
    fun actualFrameCount(): Int = (videoFrames - 1) / 4 * 4 + 1
}

class ImageClient internal constructor(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val resolver: ModelResolver,
) : AutoCloseable {
    @Volatile
    private var lastGenerationMetrics: GenerationMetrics? = null

    private val generationMutex = Mutex()

    @Volatile
    private var activeModel: StableDiffusion? = null

    /**
     * Generate a single bitmap from text.
     *
     * This client resolves and loads the requested model for the operation, while advanced callers
     * can use [StableDiffusion] directly when they need to hold a warmed runtime across requests.
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution, loading, or native
     * generation fails.
     */
    suspend fun generate(
        params: ImageGenerationRequest,
    ): Bitmap =
        generationMutex.withLock {
            lastGenerationMetrics = null
            val modelPath = resolver.resolve(context, params.model ?: config.models.image)
            val flashAttn =
                FlashAttentionHelper.shouldUseFlashAttention(
                    width = params.width,
                    height = params.height,
                    forceEnable = if (params.flashAttention) null else false,
                )
            val model =
                StableDiffusion.load(
                    context = context,
                    modelPath = modelPath.absolutePath,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = !config.preferPerformanceMode,
                    sequentialLoad = if (params.forceSequentialLoad) true else null,
                    forceVulkan = config.preferPerformanceMode && LLMEdge.isVulkanAvailable(),
                    preferPerformanceMode = config.preferPerformanceMode,
                    flashAttn = flashAttn,
                    vaeDecodeOnly = true,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                )
            activeModel = model
            try {
                val easyCache = resolveEasyCache(model, params.easyCache)
                model.txt2img(
                    GenerateParams(
                        prompt = params.prompt,
                        negative = params.negative,
                        width = params.width,
                        height = params.height,
                        steps = params.steps,
                        cfgScale = params.cfgScale,
                        seed = params.seed,
                        easyCacheParams = easyCache,
                    ),
                ).also {
                    lastGenerationMetrics = model.getLastGenerationMetrics()
                }
            } finally {
                activeModel = null
                model.close()
            }
        }

    /**
     * Stream progress and final frames for text-to-video generation.
     *
     * Cancel the returned flow collection to stop the active generation.
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution, loading, or native
     * generation fails.
     */
    fun generateVideo(
        params: VideoGenerationRequest,
    ): Flow<GenerationStreamEvent> =
        callbackFlow {
            val job =
                scope.coroutineScope.launch {
                    try {
                        val frames =
                            generationMutex.withLock {
                                if (params.forceSequentialLoad) {
                                    generateVideoSequentially(params) { message, current, total ->
                                        trySend(
                                            GenerationStreamEvent.Progress(
                                                ProgressEvent.Step(message, current, total),
                                            ),
                                        )
                                    }
                                } else {
                                    generateVideoDirect(params) { message, current, total ->
                                        trySend(
                                            GenerationStreamEvent.Progress(
                                                ProgressEvent.Step(message, current, total),
                                            ),
                                        )
                                    }
                                }
                            }
                        trySend(GenerationStreamEvent.Completed(frames))
                        close()
                    } catch (t: Throwable) {
                        close(t)
                    }
                }
            awaitClose {
                job.cancel()
                cancelGeneration()
            }
        }

    /** Request cancellation for the active generation, if any. */
    fun cancelGeneration() {
        activeModel?.cancelGeneration()
    }

    fun getLastGenerationMetrics(): GenerationMetrics? = lastGenerationMetrics

    private suspend fun generateVideoDirect(
        params: VideoGenerationRequest,
        onProgress: ((String, Int, Int) -> Unit)? = null,
    ): List<Bitmap> {
        lastGenerationMetrics = null
        val modelPath = resolver.resolve(context, params.model ?: config.models.video.diffusion)
        val taesdPath = params.taehv?.let { resolver.resolve(context, it).absolutePath }
        val vaePath = if (taesdPath == null) resolveVideoVae(params)?.absolutePath else null
        val textEncoderPath = resolveVideoTextEncoder(params)?.absolutePath
        val usingCustomTae = taesdPath != null
        val tryVulkan = !usingCustomTae && (config.preferPerformanceMode && LLMEdge.isVulkanAvailable())

        suspend fun loadAndGenerate(forceVulkan: Boolean, allowVulkan: Boolean = true): List<Bitmap> {
            val model =
                StableDiffusion.load(
                    context = context,
                    modelPath = modelPath.absolutePath,
                    vaePath = vaePath,
                    t5xxlPath = textEncoderPath,
                    taesdPath = taesdPath,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = params.forceSequentialLoad || usingCustomTae || !config.preferPerformanceMode,
                    keepClipOnCpu = usingCustomTae || !config.preferPerformanceMode,
                    keepVaeOnCpu = usingCustomTae || !config.preferPerformanceMode,
                    flashAttn = params.flashAttention,
                    vaeDecodeOnly = params.initImage == null,
                    sequentialLoad = if (params.forceSequentialLoad) true else null,
                    allowVulkan = allowVulkan,
                    forceVulkan = forceVulkan,
                    preferPerformanceMode = config.preferPerformanceMode,
                    flowShift = params.flowShift,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                )
            activeModel = model
            try {
                val easyCache = resolveEasyCache(model, params.easyCache)
                return model.txt2vid(
                    params =
                        VideoGenerateParams(
                            prompt = params.prompt,
                            negative = params.negative,
                            width = params.width,
                            height = params.height,
                            videoFrames = params.videoFrames,
                            steps = params.steps,
                            cfgScale = params.cfgScale,
                            seed = params.seed,
                            initImage = params.initImage,
                            strength = params.strength,
                            sampleMethod = params.sampleMethod,
                            scheduler = params.scheduler,
                            easyCacheParams = easyCache,
                        ),
                    onProgress =
                        VideoProgressCallback { step, totalSteps, currentFrame, totalFrames, _ ->
                            onProgress?.invoke(
                                "Generating frame $currentFrame/$totalFrames",
                                step,
                                totalSteps,
                            )
                        },
                ).also {
                    lastGenerationMetrics = model.getLastGenerationMetrics()
                }
            } finally {
                activeModel = null
                model.close()
            }
        }

        return try {
            loadAndGenerate(forceVulkan = tryVulkan)
        } catch (t: Throwable) {
            if (tryVulkan && isVulkanDeviceLost(t)) {
                AndroidLogAdapter.w(
                    "ImageClient",
                    "Vulkan device lost during video generation; retrying once with Vulkan disabled (CPU backend)",
                )
                loadAndGenerate(forceVulkan = false, allowVulkan = false)
            } else {
                throw t
            }
        }
    }

    private fun isVulkanDeviceLost(t: Throwable): Boolean {
        var cur: Throwable? = t
        while (cur != null) {
            val msg = cur.message
            if (msg != null && (
                    msg.contains("ErrorDeviceLost", ignoreCase = true) ||
                        msg.contains("VK_ERROR_DEVICE_LOST", ignoreCase = true) ||
                        msg.contains("DeviceLost", ignoreCase = true)
                )
            ) {
                return true
            }
            cur = cur.cause
        }
        return false
    }

    private suspend fun generateVideoSequentially(
        params: VideoGenerationRequest,
        onProgress: ((String, Int, Int) -> Unit)? = null,
    ): List<Bitmap> {
        lastGenerationMetrics = null
        val modelPath = resolver.resolve(context, params.model ?: config.models.video.diffusion)
        val taesdPath = params.taehv?.let { resolver.resolve(context, it).absolutePath }
        val usingCustomTae = taesdPath != null
        val vaePath = if (taesdPath == null) resolveRequiredVideoVae(params).absolutePath else null
        val textEncoderPath = resolveRequiredVideoTextEncoder(params).absolutePath
        val tryVulkan = !usingCustomTae && (config.preferPerformanceMode && LLMEdge.isVulkanAvailable())

        onProgress?.invoke("Loading text encoder", 0, params.steps)
        val t5Model =
            StableDiffusion.load(
                context = context,
                modelPath = textEncoderPath,
                vaePath = null,
                t5xxlPath = null,
                nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                forceVulkan = tryVulkan,
                preferPerformanceMode = config.preferPerformanceMode,
                flashAttn = params.flashAttention,
            )

        val cond: PrecomputedCondition?
        val uncond: PrecomputedCondition?
        try {
            onProgress?.invoke("Precomputing prompt conditioning", 0, params.steps)
            cond =
                t5Model.precomputeCondition(
                    prompt = params.prompt,
                    negative = params.negative,
                    width = params.width,
                    height = params.height,
                )
            uncond =
                if (params.cfgScale != 1.0f) {
                    t5Model.precomputeCondition(
                        prompt = params.negative,
                        negative = "",
                        width = params.width,
                        height = params.height,
                    )
                } else {
                    null
                }
        } finally {
            t5Model.close()
        }

        suspend fun loadAndGenerate(forceVulkan: Boolean, allowVulkan: Boolean = true): List<Bitmap> {
            onProgress?.invoke("Loading diffusion model", 0, params.steps)
            val diffusionModel =
                StableDiffusion.load(
                    context = context,
                    modelPath = modelPath.absolutePath,
                    vaePath = vaePath,
                    t5xxlPath = null,
                    taesdPath = taesdPath,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = true,
                    keepClipOnCpu = true,
                    keepVaeOnCpu = true,
                    allowVulkan = allowVulkan,
                    forceVulkan = forceVulkan,
                    preferPerformanceMode = config.preferPerformanceMode,
                    flashAttn = params.flashAttention,
                    vaeDecodeOnly = params.initImage == null,
                    flowShift = params.flowShift,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                )
            activeModel = diffusionModel
            try {
                val easyCache = resolveEasyCache(diffusionModel, params.easyCache)
                return diffusionModel.txt2VidWithPrecomputedCondition(
                    params =
                        VideoGenerateParams(
                            prompt = params.prompt,
                            negative = params.negative,
                            width = params.width,
                            height = params.height,
                            videoFrames = params.videoFrames,
                            steps = params.steps,
                            cfgScale = params.cfgScale,
                            seed = params.seed,
                            initImage = params.initImage,
                            strength = params.strength,
                            sampleMethod = params.sampleMethod,
                            scheduler = params.scheduler,
                            easyCacheParams = easyCache,
                        ),
                    cond = cond,
                    uncond = uncond,
                    onProgress =
                        VideoProgressCallback { step, totalSteps, currentFrame, totalFrames, _ ->
                            onProgress?.invoke(
                                "Generating frame $currentFrame/$totalFrames",
                                step,
                                totalSteps,
                            )
                        },
                ).also {
                    lastGenerationMetrics = diffusionModel.getLastGenerationMetrics()
                }
            } finally {
                activeModel = null
                diffusionModel.close()
            }
        }

        return try {
            loadAndGenerate(forceVulkan = tryVulkan)
        } catch (t: Throwable) {
            if (tryVulkan && isVulkanDeviceLost(t)) {
                AndroidLogAdapter.w(
                    "ImageClient",
                    "Vulkan device lost during video generation; retrying once with Vulkan disabled (CPU backend)",
                )
                loadAndGenerate(forceVulkan = false, allowVulkan = false)
            } else {
                throw t
            }
        }
    }

    private suspend fun resolveVideoVae(params: VideoGenerationRequest): java.io.File? {
        val spec = params.vae ?: config.models.video.vae
        return spec.let { resolver.resolve(context, it) }
    }

    private suspend fun resolveRequiredVideoVae(params: VideoGenerationRequest): java.io.File {
        return requireNotNull(resolveVideoVae(params)) {
            "Video generation requires either a VAE model or a TAEHV/TAESD override."
        }
    }

    private suspend fun resolveVideoTextEncoder(params: VideoGenerationRequest): java.io.File? {
        val spec = params.textEncoder ?: config.models.video.textEncoder
        return spec.let { resolver.resolve(context, it) }
    }

    private suspend fun resolveRequiredVideoTextEncoder(params: VideoGenerationRequest): java.io.File {
        return requireNotNull(resolveVideoTextEncoder(params)) {
            "Sequential video generation requires a text encoder model."
        }
    }

    private fun resolveEasyCache(
        model: StableDiffusion,
        requested: EasyCacheParams,
    ): EasyCacheParams =
        if (model.isEasyCacheSupported()) {
            requested.copy(enabled = true)
        } else {
            requested.copy(enabled = false)
        }

    override fun close() {
        cancelGeneration()
        activeModel = null
    }
}
