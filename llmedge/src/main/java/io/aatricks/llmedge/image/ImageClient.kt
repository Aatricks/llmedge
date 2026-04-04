package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.ClientBootstrap
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.core.runtime.executeWithRuntimeRetry
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationPhase
import io.aatricks.llmedge.image.diffusion.ImageGenerationTraceEvent
import io.aatricks.llmedge.image.diffusion.ImageRequestMetrics
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
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
    private val resolver: ModelRepository,
    private val ownedBootstrap: ClientBootstrapContext? = null,
) : AutoCloseable {
    companion object {
        private const val LOG_TAG = "ImageClient"

        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): ImageClient =
            ClientBootstrap.createOwned(context, scope, config.text.promptThreads) { bootstrap ->
                ImageClient(
                    context = bootstrap.appContext,
                    scope = bootstrap.edgeScope,
                    config = config,
                    resolver = modelRepository,
                    ownedBootstrap = bootstrap,
                )
            }

        internal fun resetVideoVulkanBlacklistForTests() {
            BackendRuntimePolicy.resetForTests()
        }
    }

    @Volatile
    private var lastGenerationMetrics: GenerationMetrics? = null

    @Volatile
    private var lastImageRequestTrace: List<ImageGenerationTraceEvent> = emptyList()

    private val runtimePool = createDiffusionRuntimePool(context, scope, config, resolver)
    private val generationMutex = Mutex()
    private val imageRequestIds = AtomicLong(0L)

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
            lastImageRequestTrace = emptyList()
            val request = ImageRuntimeRequestPlanner.imageRequest(params, config)
            val requestId = imageRequestIds.incrementAndGet()
            AndroidLogAdapter.i(
                LOG_TAG,
                "Image request entering runtime acquisition: requestId=$requestId size=${params.width}x${params.height} steps=${params.steps}",
            )
            executeWithRuntimeRetry(
                spec = request.spec,
                options = request.options,
                retryMessage = "Retrying image generation on the next backend after a backend-specific failure for",
            ) { acquired ->
                val runtime = acquired.runtime
                AndroidLogAdapter.i(
                    LOG_TAG,
                    "Image request runtime acquired: requestId=$requestId cacheHit=${acquired.acquire.cacheHit} backend=${acquired.acquire.backend} loadMs=${acquired.acquire.modelLoadTimeMs}",
                )
                runtime.mutex.withLock {
                    AndroidLogAdapter.i(
                        LOG_TAG,
                        "Image request runtime mutex acquired: requestId=$requestId backend=${acquired.acquire.backend}",
                    )
                    withActiveModel(runtime.model) { model ->
                        AndroidLogAdapter.i(
                            LOG_TAG,
                            "Image request active model entered: requestId=$requestId",
                        )
                        model.beginImageRequestTrace(requestId)
                        model.traceImagePhase(
                            ImageGenerationPhase.REQUESTED,
                            "promptChars=${params.prompt.length} size=${params.width}x${params.height} steps=${params.steps}",
                        )
                        model.traceImagePhase(
                            ImageGenerationPhase.RUNTIME_ACQUIRED,
                            "cacheHit=${acquired.acquire.cacheHit} acquireMs=${acquired.acquire.acquireTimeMs} loadMs=${acquired.acquire.modelLoadTimeMs} backend=${acquired.acquire.backend.name}",
                        )
                        try {
                            val easyCache = resolveEasyCache(model, params.easyCache)
                            model.traceImagePhase(
                                ImageGenerationPhase.MODEL_READY,
                                "flash=${runtime.flashAttnEnabled} easyCache=${easyCache.enabled}",
                            )
                            model.traceImagePhase(
                                ImageGenerationPhase.TXT2IMG_ENTER,
                                "ImageClient dispatching to StableDiffusion.txt2img",
                            )
                            val generationStartNanos = System.nanoTime()
                            val bitmap =
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
                                )
                            val generateMs = elapsedMillis(generationStartNanos)
                            val requestMetrics =
                                ImageRequestMetrics(
                                    runtimeAcquireMs = acquired.acquire.acquireTimeMs,
                                    modelLoadMs = acquired.acquire.modelLoadTimeMs,
                                    generateMs = generateMs,
                                    cacheHit = acquired.acquire.cacheHit,
                                    backend = acquired.acquire.backend.name,
                                    flashAttentionEnabled = runtime.flashAttnEnabled,
                                    easyCacheEnabled = easyCache.enabled,
                                    width = params.width,
                                    height = params.height,
                                    steps = params.steps,
                                )
                            val baseMetrics =
                                model.getLastGenerationMetrics()
                                    ?: fallbackImageMetrics(
                                        runtime = runtime,
                                        params = params,
                                        generateMs = generateMs,
                                    )
                            lastGenerationMetrics = baseMetrics.withImageRequestMetrics(requestMetrics)
                            logImageRequestMetrics(acquired.acquire.keyPrefix, requestMetrics)
                            bitmap
                        } catch (cancelled: CancellationException) {
                            lastGenerationMetrics = null
                            model.traceImagePhase(
                                ImageGenerationPhase.CANCELLED,
                                "ImageClient observed cancellation",
                                cancelled,
                            )
                            throw cancelled
                        } catch (t: Throwable) {
                            lastGenerationMetrics = null
                            model.traceImagePhase(
                                ImageGenerationPhase.FAILED,
                                "ImageClient observed failure: ${t.message}",
                                t,
                            )
                            throw t
                        } finally {
                            lastImageRequestTrace = model.getLastImageRequestTraceForTests()
                            AndroidLogAdapter.i(
                                LOG_TAG,
                                "Image request active model exit: requestId=$requestId phases=${lastImageRequestTrace.size}",
                            )
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

    internal fun getLastImageRequestTraceForTests(): List<ImageGenerationTraceEvent> = lastImageRequestTrace

    private suspend fun generateVideoDirect(
        params: VideoGenerationRequest,
        onProgress: ((String, Int, Int) -> Unit)? = null,
    ): List<Bitmap> {
        lastGenerationMetrics = null
        val request = ImageRuntimeRequestPlanner.directVideoRequest(params, config)
        return executeWithRuntimeRetry(
            spec = request.spec,
            options = request.options,
            retryMessage = "Retrying video generation on the next backend after a backend-specific failure for",
        ) { acquired ->
            acquired.runtime.mutex.withLock {
                withActiveModel(acquired.runtime.model) { model ->
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
        val conditioningRequest = ImageRuntimeRequestPlanner.sequentialVideoConditioningRequest(params, config)
        val conditioning =
            executeWithRuntimeRetry(
                spec = conditioningRequest.spec,
                options = conditioningRequest.options,
                retryMessage = "Retrying sequential video text conditioning on the next backend after a backend-specific failure for",
            ) { acquired ->
                acquired.runtime.mutex.withLock {
                    withActiveModel(acquired.runtime.model) { model ->
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

        val diffusionRequest = ImageRuntimeRequestPlanner.sequentialVideoDiffusionRequest(params, config)
        return executeWithRuntimeRetry(
            spec = diffusionRequest.spec,
            options = diffusionRequest.options,
            retryMessage = "Retrying sequential video diffusion on the next backend after a backend-specific failure for",
        ) { acquired ->
            acquired.runtime.mutex.withLock {
                withActiveModel(acquired.runtime.model) { model ->
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
    ): EasyCacheParams {
        if (!requested.enabled) {
            return requested.copy(enabled = false)
        }
        return if (model.isEasyCacheSupported()) {
            requested
        } else {
            AndroidLogAdapter.w(
                LOG_TAG,
                "EasyCache requested but unsupported for the current model; disabling it for this request",
            )
            requested.copy(enabled = false)
        }
    }

    override fun close() {
        ClientBootstrap.close(ownedBootstrap) {
            cancelGeneration()
            activeModel = null
            runtimePool.close()
        }
    }

    private suspend fun <T> executeWithRuntimeRetry(
        spec: DiffusionRuntimeSpec,
        options: DiffusionLoadOptions,
        retryMessage: String,
        execute: suspend (AcquiredDiffusionRuntime) -> T,
    ): T =
        runtimePool.executeWithRuntimeRetry(
            spec = spec,
            options = options,
            onRetry = { _, _ -> AndroidLogAdapter.w(LOG_TAG, "$retryMessage '${spec.model.cacheKey}'") },
        ) { execution ->
            AndroidLogAdapter.i(
                LOG_TAG,
                "Runtime acquire completed: role=${spec.role} backend=${execution.acquire.backend} cacheHit=${execution.acquire.cacheHit} loadMs=${execution.acquire.modelLoadTimeMs}",
            )
            execute(AcquiredDiffusionRuntime(execution.runtime, execution.acquire))
        }

    private fun fallbackImageMetrics(
        runtime: ManagedDiffusionModel,
        params: ImageGenerationRequest,
        generateMs: Long,
    ): GenerationMetrics {
        val totalSeconds = generateMs / 1000f
        return GenerationMetrics(
            totalTimeSeconds = totalSeconds,
            framesPerSecond = if (totalSeconds > 0f) 1f / totalSeconds else 0f,
            timePerStep = if (params.steps > 0) totalSeconds / params.steps else 0f,
            peakMemoryUsageMb = 0L,
            vulkanEnabled = runtime.backend == ComputeBackend.VULKAN,
            frameConversionTimeSeconds = 0f,
        )
    }

    private fun logImageRequestMetrics(
        cacheKeyPrefix: String,
        metrics: ImageRequestMetrics,
    ) {
        AndroidLogAdapter.i(
            LOG_TAG,
            "Image request metrics: cacheHit=${metrics.cacheHit}, loadMs=${metrics.modelLoadMs}, acquireMs=${metrics.runtimeAcquireMs}, " +
                "generateMs=${metrics.generateMs}, totalMs=${metrics.totalWallTimeMs}, flash=${metrics.flashAttentionEnabled}, " +
                "easyCache=${metrics.easyCacheEnabled}, size=${metrics.width}x${metrics.height}, steps=${metrics.steps}",
        )
        AndroidLogAdapter.d(
            LOG_TAG,
            "Image runtime debug: key='$cacheKeyPrefix', backend=${metrics.backend}",
        )
    }

    private fun elapsedMillis(startNanos: Long): Long =
        ((System.nanoTime() - startNanos) / 1_000_000L).coerceAtLeast(0L)

    private data class AcquiredDiffusionRuntime(
        val runtime: ManagedDiffusionModel,
        val acquire: io.aatricks.llmedge.core.runtime.RuntimeAcquireResult<ManagedDiffusionModel>,
    )

    private suspend fun <T> withActiveModel(
        model: StableDiffusion,
        execute: suspend (StableDiffusion) -> T,
    ): T {
        activeModel = model
        AndroidLogAdapter.i(LOG_TAG, "withActiveModel enter: handle=${System.identityHashCode(model)}")
        return try {
            execute(model)
        } finally {
            model.clearImageRequestTrace()
            if (activeModel === model) {
                activeModel = null
            }
            AndroidLogAdapter.i(LOG_TAG, "withActiveModel exit: handle=${System.identityHashCode(model)}")
        }
    }
}
