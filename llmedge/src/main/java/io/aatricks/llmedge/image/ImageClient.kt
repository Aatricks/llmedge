package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.runtime.FlashAttentionHelper
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
    companion object {
        private const val LOG_TAG = "ImageClient"

        internal fun resetVideoVulkanBlacklistForTests() {
            BackendRuntimePolicy.resetForTests()
        }
    }

    @Volatile
    private var lastGenerationMetrics: GenerationMetrics? = null

    private val runtimePool = createDiffusionRuntimePool(context, scope, config, resolver)
    private val generationMutex = Mutex()

    @Volatile
    private var activeModel: StableDiffusion? = null

    private data class RuntimeRequest(
        val spec: DiffusionRuntimeSpec,
        val options: DiffusionLoadOptions,
    )

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
            val request = imageRuntimeRequest(params)
            executeWithRuntimeRetry(
                request = request,
                retryMessage = "Retrying image generation on the next backend after a backend-specific failure for",
            ) { runtime ->
                runtime.mutex.withLock {
                    withActiveModel(runtime.model) { model ->
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
                    }
                }
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
        val request = directVideoRuntimeRequest(params)
        return executeWithRuntimeRetry(
            request = request,
            retryMessage = "Retrying video generation on the next backend after a backend-specific failure for",
        ) { runtime ->
            runtime.mutex.withLock {
                withActiveModel(runtime.model) { model ->
                    val easyCache = resolveEasyCache(model, params.easyCache)
                    model.txt2vid(
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
                }
            }
        }
    }

    private suspend fun generateVideoSequentially(
        params: VideoGenerationRequest,
        onProgress: ((String, Int, Int) -> Unit)? = null,
    ): List<Bitmap> {
        lastGenerationMetrics = null
        val conditioningRequest = sequentialVideoConditioningRequest(params)
        val conditioning =
            executeWithRuntimeRetry(
                request = conditioningRequest,
                retryMessage = "Retrying sequential video text conditioning on the next backend after a backend-specific failure for",
            ) { runtime ->
                runtime.mutex.withLock {
                    withActiveModel(runtime.model) { model ->
                        onProgress?.invoke("Loading text encoder", 0, params.steps)
                        onProgress?.invoke("Precomputing prompt conditioning", 0, params.steps)
                        val cond =
                            model.precomputeCondition(
                                prompt = params.prompt,
                                negative = params.negative,
                                width = params.width,
                                height = params.height,
                            )
                        val uncond =
                            if (params.cfgScale != 1.0f) {
                                model.precomputeCondition(
                                    prompt = params.negative,
                                    negative = "",
                                    width = params.width,
                                    height = params.height,
                                )
                            } else {
                                null
                            }
                        cond to uncond
                    }
                }
            }

        val diffusionRequest = sequentialVideoDiffusionRequest(params)
        return executeWithRuntimeRetry(
            request = diffusionRequest,
            retryMessage = "Retrying sequential video diffusion on the next backend after a backend-specific failure for",
        ) { runtime ->
            runtime.mutex.withLock {
                withActiveModel(runtime.model) { model ->
                    onProgress?.invoke("Loading diffusion model", 0, params.steps)
                    val easyCache = resolveEasyCache(model, params.easyCache)
                    model.txt2VidWithPrecomputedCondition(
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
                        cond = conditioning.first,
                        uncond = conditioning.second,
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
                }
            }
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
        runtimePool.close()
    }

    private fun imageRuntimeRequest(params: ImageGenerationRequest): RuntimeRequest {
        val flashAttn =
            FlashAttentionHelper.shouldUseFlashAttention(
                width = params.width,
                height = params.height,
                forceEnable = if (params.flashAttention) null else false,
            )
        return RuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.IMAGE,
                    model = params.model ?: config.models.image,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.IMAGE,
                    allowGpu = config.preferPerformanceMode,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = !config.preferPerformanceMode,
                    keepClipOnCpu = false,
                    keepVaeOnCpu = false,
                    flashAttn = flashAttn,
                    vaeDecodeOnly = true,
                    sequentialLoad = if (params.forceSequentialLoad) true else null,
                    preferPerformanceMode = config.preferPerformanceMode,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                ),
        )
    }

    private fun directVideoRuntimeRequest(params: VideoGenerationRequest): RuntimeRequest {
        val usingCustomTae = params.taehv != null
        return RuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.VIDEO,
                    model = params.model ?: config.models.video.diffusion,
                    vae = if (usingCustomTae) null else (params.vae ?: config.models.video.vae),
                    textEncoder = params.textEncoder ?: config.models.video.textEncoder,
                    taehv = params.taehv,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.VIDEO,
                    allowGpu = config.preferPerformanceMode && !usingCustomTae,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = usingCustomTae || !config.preferPerformanceMode,
                    keepClipOnCpu = usingCustomTae || !config.preferPerformanceMode,
                    keepVaeOnCpu = usingCustomTae || !config.preferPerformanceMode,
                    flashAttn = params.flashAttention,
                    vaeDecodeOnly = params.initImage == null,
                    preferPerformanceMode = config.preferPerformanceMode,
                    flowShift = params.flowShift,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                ),
        )
    }

    private fun sequentialVideoConditioningRequest(params: VideoGenerationRequest): RuntimeRequest {
        val usingCustomTae = params.taehv != null
        return RuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.VIDEO_TEXT_ENCODER,
                    model = params.textEncoder ?: config.models.video.textEncoder,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.VIDEO,
                    allowGpu = config.preferPerformanceMode && !usingCustomTae,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
                    offloadToCpu = true,
                    keepClipOnCpu = true,
                    keepVaeOnCpu = true,
                    flashAttn = params.flashAttention,
                    preferPerformanceMode = config.preferPerformanceMode,
                ),
        )
    }

    private fun sequentialVideoDiffusionRequest(params: VideoGenerationRequest): RuntimeRequest {
        val usingCustomTae = params.taehv != null
        return RuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.VIDEO,
                    model = params.model ?: config.models.video.diffusion,
                    vae = if (usingCustomTae) null else (params.vae ?: config.models.video.vae),
                    taehv = params.taehv,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.VIDEO,
                    allowGpu = config.preferPerformanceMode && !usingCustomTae,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = true,
                    keepClipOnCpu = true,
                    keepVaeOnCpu = true,
                    flashAttn = params.flashAttention,
                    vaeDecodeOnly = params.initImage == null,
                    preferPerformanceMode = config.preferPerformanceMode,
                    flowShift = params.flowShift,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                ),
        )
    }

    private suspend fun <T> executeWithRuntimeRetry(
        request: RuntimeRequest,
        retryMessage: String,
        execute: suspend (ManagedDiffusionModel) -> T,
    ): T {
        while (true) {
            val runtime = runtimePool.acquire(request.spec, request.options)
            try {
                return execute(runtime)
            } catch (error: Throwable) {
                val blacklisted =
                    runtimePool.recordBackendFailureIfNeeded(
                        request.spec,
                        request.options,
                        runtime,
                        error,
                    )
                if (!blacklisted) {
                    throw error
                }
                AndroidLogAdapter.w(LOG_TAG, "$retryMessage '${request.spec.model.cacheKey}'")
            }
        }
    }

    private suspend fun <T> withActiveModel(
        model: StableDiffusion,
        execute: suspend (StableDiffusion) -> T,
    ): T {
        activeModel = model
        return try {
            execute(model)
        } finally {
            if (activeModel === model) {
                activeModel = null
            }
        }
    }
}
