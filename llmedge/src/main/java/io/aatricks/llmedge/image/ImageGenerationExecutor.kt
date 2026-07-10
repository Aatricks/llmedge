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
    private val phaseListener: DiffusionPhaseListener? = null,
) {
    suspend fun generate(params: ImageGenerationRequest): Bitmap {
        if (params.sequential && params.splitDiffusionModel && params.textEncoder != null) {
            return generateSequential(params)
        }
        return generateDirect(params)
    }

    /**
     * FLUX.2 sequential low-memory generation: phase 1 loads only the Qwen3 encoder to precompute
     * the conditioning (then frees it), phase 2 loads only the DiT to generate. Peak RAM is the
     * larger phase, not the sum.
     */
    private suspend fun generateSequential(params: ImageGenerationRequest): Bitmap =
        generationMutex.withLock {
            state.resetForRequest()
            val plan = ImageRuntimeRequestPlanner.imageSequentialPlan(params, config)
            val requestId = imageRequestIds.incrementAndGet()
            AndroidLogAdapter.i(logTag, "FLUX.2 sequential image request: requestId=$requestId")

            val cond =
                requestExecutor.withRuntimeModel(
                    spec = plan.conditioningRequest.spec,
                    options = plan.conditioningRequest.options,
                    retryMessage = "Retrying FLUX.2 sequential conditioning on the next backend for",
                ) { model, _, acquire ->
                    phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.GENERATING, acquire.backend.name)
                    model.precomputeCondition(params.prompt, "", params.width, params.height, -1)
                }

            // Free the encoder before the DiT loads: cache eviction only runs inside put()
            // AFTER the phase-2 load completes (and closes asynchronously), so without this
            // the peak is encoder+DiT — the sum the sequential mode exists to avoid.
            requestExecutor.invalidateRuntime(
                plan.conditioningRequest.spec,
                plan.conditioningRequest.options,
            )
            AndroidLogAdapter.i(logTag, "FLUX.2 sequential: encoder runtime invalidated before diffusion phase")

            requestExecutor.withRuntimeModel(
                spec = plan.diffusionRequest.spec,
                options = plan.diffusionRequest.options,
                retryMessage = "Retrying FLUX.2 sequential diffusion on the next backend for",
            ) { model, _, acquire ->
                phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.GENERATING, acquire.backend.name)
                phaseListener?.let { listener ->
                    model.setProgressCallback { step, totalSteps, _, _, _ ->
                        listener.onStep(step, totalSteps)
                    }
                }
                val rgb =
                    model.txt2ImgWithPrecomputedCondition(
                        prompt = params.prompt,
                        negative = "",
                        width = params.width,
                        height = params.height,
                        steps = params.steps,
                        cfg = params.cfgScale,
                        seed = params.seed,
                        cond = cond,
                        uncond = null,
                    ) ?: throw IllegalStateException("FLUX.2 sequential generation returned null")
                rgbToBitmap(rgb, params.width, params.height)
            }
        }

    private fun rgbToBitmap(rgb: ByteArray, width: Int, height: Int): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val px = IntArray(width * height)
        for (i in 0 until width * height) {
            val r = rgb[i * 3].toInt() and 0xFF
            val g = rgb[i * 3 + 1].toInt() and 0xFF
            val b = rgb[i * 3 + 2].toInt() and 0xFF
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bmp.setPixels(px, 0, width, 0, 0, width, height)
        return bmp
    }

    private suspend fun generateDirect(params: ImageGenerationRequest): Bitmap =
        generationMutex.withLock {
            state.resetForRequest()
            val request = ImageRuntimeRequestPlanner.imageRequest(params, config)
            val requestId = imageRequestIds.incrementAndGet()
            AndroidLogAdapter.i(
                logTag,
                "Image request entering runtime acquisition: requestId=$requestId size=${params.width}x${params.height} steps=${params.steps}",
            )
            requestExecutor.withRuntimeModel(
                spec = request.spec,
                options = request.options,
                retryMessage = "Retrying image generation on the next backend after a backend-specific failure for",
            ) { model, runtime, acquire ->
                AndroidLogAdapter.i(
                    logTag,
                    "Image request runtime acquired: requestId=$requestId cacheHit=${acquire.cacheHit} backend=${acquire.backend} loadMs=${acquire.modelLoadTimeMs}",
                )
                AndroidLogAdapter.i(
                    logTag,
                    "Image request runtime mutex acquired: requestId=$requestId backend=${acquire.backend}",
                )
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
                    "cacheHit=${acquire.cacheHit} acquireMs=${acquire.acquireTimeMs} loadMs=${acquire.modelLoadTimeMs} backend=${acquire.backend.name}",
                )
                try {
                    val easyCache =
                        model.resolveEasyCacheParams(params.easyCache) {
                            AndroidLogAdapter.w(
                                logTag,
                                "EasyCache requested but unsupported for the current model; disabling it for this request",
                            )
                        }
                    model.traceImagePhase(
                        ImageGenerationPhase.MODEL_READY,
                        "flash=${runtime.flashAttnEnabled} easyCache=${easyCache.enabled}",
                    )
                    model.traceImagePhase(
                        ImageGenerationPhase.TXT2IMG_ENTER,
                        "ImageClient dispatching to StableDiffusion.txt2img",
                    )
                    phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.GENERATING, acquire.backend.name)
                    phaseListener?.let { listener ->
                        model.setProgressCallback { step, totalSteps, _, _, _ ->
                            listener.onStep(step, totalSteps)
                        }
                    }
                    val generationStartNanos = System.nanoTime()
                    val bitmap =
                        try {
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
                        } finally {
                            if (phaseListener != null) model.setProgressCallback(null)
                        }
                    val generateMs = elapsedMillis(generationStartNanos)
                    val requestMetrics =
                        ImageRequestMetrics(
                            runtimeAcquireMs = acquire.acquireTimeMs,
                            modelLoadMs = acquire.modelLoadTimeMs,
                            generateMs = generateMs,
                            cacheHit = acquire.cacheHit,
                            backend = acquire.backend.name,
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
                    logImageRequestMetrics(acquire.keyPrefix, requestMetrics)
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
