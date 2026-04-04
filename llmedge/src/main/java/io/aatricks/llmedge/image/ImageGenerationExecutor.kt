package io.aatricks.llmedge.image

import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationPhase
import io.aatricks.llmedge.image.diffusion.ImageRequestMetrics
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.runtime.ComputeBackend
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

internal class ImageGenerationExecutor(
    private val config: LLMEdgeConfig,
    private val generationMutex: Mutex,
    private val imageRequestIds: AtomicLong,
    private val state: ImageClientState,
    private val requestExecutor: DiffusionRequestExecutor,
    private val logTag: String,
) {
    suspend fun generate(params: ImageGenerationRequest): Bitmap =
        generationMutex.withLock {
            state.resetForRequest()
            val request = ImageRuntimeRequestPlanner.imageRequest(params, config)
            val requestId = imageRequestIds.incrementAndGet()
            AndroidLogAdapter.i(
                logTag,
                "Image request entering runtime acquisition: requestId=$requestId size=${params.width}x${params.height} steps=${params.steps}",
            )
            requestExecutor.executeWithRuntimeRetry(
                spec = request.spec,
                options = request.options,
                retryMessage = "Retrying image generation on the next backend after a backend-specific failure for",
            ) { acquired ->
                val runtime = acquired.runtime
                AndroidLogAdapter.i(
                    logTag,
                    "Image request runtime acquired: requestId=$requestId cacheHit=${acquired.acquire.cacheHit} backend=${acquired.acquire.backend} loadMs=${acquired.acquire.modelLoadTimeMs}",
                )
                runtime.mutex.withLock {
                    AndroidLogAdapter.i(
                        logTag,
                        "Image request runtime mutex acquired: requestId=$requestId backend=${acquired.acquire.backend}",
                    )
                    requestExecutor.withActiveModel(runtime.model) { model ->
                        AndroidLogAdapter.i(
                            logTag,
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
                            state.lastGenerationMetrics =
                                baseMetrics.withImageRequestMetrics(requestMetrics)
                            logImageRequestMetrics(acquired.acquire.keyPrefix, requestMetrics)
                            bitmap
                        } catch (cancelled: CancellationException) {
                            state.lastGenerationMetrics = null
                            model.traceImagePhase(
                                ImageGenerationPhase.CANCELLED,
                                "ImageClient observed cancellation",
                                cancelled,
                            )
                            throw cancelled
                        } catch (t: Throwable) {
                            state.lastGenerationMetrics = null
                            model.traceImagePhase(
                                ImageGenerationPhase.FAILED,
                                "ImageClient observed failure: ${t.message}",
                                t,
                            )
                            throw t
                        } finally {
                            state.lastImageRequestTrace = model.getLastImageRequestTraceForTests()
                            AndroidLogAdapter.i(
                                logTag,
                                "Image request active model exit: requestId=$requestId phases=${state.lastImageRequestTrace.size}",
                            )
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
                logTag,
                "EasyCache requested but unsupported for the current model; disabling it for this request",
            )
            requested.copy(enabled = false)
        }
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
            logTag,
            "Image request metrics: cacheHit=${metrics.cacheHit}, loadMs=${metrics.modelLoadMs}, acquireMs=${metrics.runtimeAcquireMs}, " +
                "generateMs=${metrics.generateMs}, totalMs=${metrics.totalWallTimeMs}, flash=${metrics.flashAttentionEnabled}, " +
                "easyCache=${metrics.easyCacheEnabled}, size=${metrics.width}x${metrics.height}, steps=${metrics.steps}",
        )
        AndroidLogAdapter.d(
            logTag,
            "Image runtime debug: key='$cacheKeyPrefix', backend=${metrics.backend}",
        )
    }

    private fun elapsedMillis(startNanos: Long): Long =
        ((System.nanoTime() - startNanos) / 1_000_000L).coerceAtLeast(0L)
}
