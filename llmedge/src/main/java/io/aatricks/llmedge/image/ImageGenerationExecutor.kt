package io.aatricks.llmedge.image

import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationPhase
import io.aatricks.llmedge.image.diffusion.ImageRequestMetrics
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
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
    private val executionPlanSelector: ImageExecutionPlanSelector,
    private val logTag: String,
    private val phaseListener: DiffusionPhaseListener? = null,
) {
    suspend fun generate(params: ImageGenerationRequest): Bitmap =
        generationMutex.withLock {
            val decision = executionPlanSelector.decide(params, config)
            if (decision.mode == ImageExecutionMode.SEQUENTIAL) {
                return@withLock generateSequential(params, decision.recipe.profile)
            }

            val directRequest = ImageRuntimeRequestPlanner.imageRequest(params, config)
            try {
                generateDirect(params, directRequest)
            } catch (error: ModelLoadException) {
                if (params.sequential != null || !decision.recipe.supportsSequential) {
                    throw error
                }
                requestExecutor.invalidateRuntime(directRequest.spec, directRequest.options)
                AndroidLogAdapter.w(logTag, "Automatic direct image load failed; retrying staged ${decision.recipe.profile} execution")
                generateSequential(params, decision.recipe.profile)
            }
        }

    /**
     * Sequential low-memory generation:
     * - For FLUX, it uses two phases: phase 1 loads only the text encoder to precompute the
     *   conditioning (then frees it), and phase 2 loads only the DiT (+VAE) to generate.
     * - For SD3, it uses three phases: phase 1 loads only the CLIP encoder to precompute conditioning
     *   (then frees it), phase 2 loads only the T5XXL encoder to precompute conditioning (then frees it),
     *   and phase 3 loads only the DiT (+VAE) to generate.
     * Peak RAM is the largest single phase, not the sum.
     */
    private suspend fun generateSequential(
        params: ImageGenerationRequest,
        profile: ImageConditioningProfile,
    ): Bitmap {
        state.resetForRequest()
        val isSd3 = profile == ImageConditioningProfile.SD3_CLIP_T5
        val isChroma = profile == ImageConditioningProfile.CHROMA_T5
        val plan = ImageRuntimeRequestPlanner.imageSequentialPlan(params, config, profile)
        val requestId = imageRequestIds.incrementAndGet()
        val modelName =
            when (profile) {
                ImageConditioningProfile.LLM -> "LLM"
                ImageConditioningProfile.SD3_CLIP_T5 -> "SD3"
                ImageConditioningProfile.CHROMA_T5 -> "Chroma"
                ImageConditioningProfile.MASKED_T5 -> "masked-T5"
                ImageConditioningProfile.NONE -> error("Sequential image execution requires a conditioning profile")
            }
        AndroidLogAdapter.i(logTag, "$modelName sequential image request: requestId=$requestId")

        val cond: PrecomputedCondition?
        val uncond: PrecomputedCondition?

        if (isSd3) {
            val condCLIP = requestExecutor.withRuntimeModel(
                spec = plan.conditioningRequest.spec,
                options = plan.conditioningRequest.options,
                retryMessage = "Retrying $modelName sequential conditioning phase A (CLIP) on the next backend for",
            ) { model, _, acquire ->
                phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.GENERATING, acquire.backend.name)
                model.precomputeCondition(params.prompt, "", params.width, params.height, -1)
            }
            val uncondCLIP = requestExecutor.withRuntimeModel(
                spec = plan.conditioningRequest.spec,
                options = plan.conditioningRequest.options,
                retryMessage = "Retrying $modelName sequential conditioning phase A (CLIP) negative on the next backend for",
            ) { model, _, _ ->
                model.precomputeCondition(params.negative, "", params.width, params.height, -1)
            }

            requestExecutor.invalidateRuntime(
                plan.conditioningRequest.spec,
                plan.conditioningRequest.options,
            )
            AndroidLogAdapter.i(logTag, "$modelName sequential: CLIP runtime invalidated before T5 phase")

            val conditioningRequest2 = plan.conditioningRequest2 ?: error("SD3 sequential requires conditioningRequest2 (T5)")

            val condT5 = requestExecutor.withRuntimeModel(
                spec = conditioningRequest2.spec,
                options = conditioningRequest2.options,
                retryMessage = "Retrying $modelName sequential conditioning phase B (T5) on the next backend for",
            ) { model, _, acquire ->
                phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.GENERATING, acquire.backend.name)
                model.precomputeCondition(params.prompt, "", params.width, params.height, -1)
            }
            val uncondT5 = requestExecutor.withRuntimeModel(
                spec = conditioningRequest2.spec,
                options = conditioningRequest2.options,
                retryMessage = "Retrying $modelName sequential conditioning phase B (T5) negative on the next backend for",
            ) { model, _, _ ->
                model.precomputeCondition(params.negative, "", params.width, params.height, -1)
            }

            requestExecutor.invalidateRuntime(
                conditioningRequest2.spec,
                conditioningRequest2.options,
            )
            AndroidLogAdapter.i(logTag, "$modelName sequential: T5 runtime invalidated before diffusion phase")

            cond = if (condCLIP != null && condT5 != null) combineSD3Condition(condCLIP, condT5) else null
            uncond = if (uncondCLIP != null && uncondT5 != null) combineSD3Condition(uncondCLIP, uncondT5) else null
        } else if (isChroma) {
            val condChroma = requestExecutor.withRuntimeModel(
                spec = plan.conditioningRequest.spec,
                options = plan.conditioningRequest.options,
                retryMessage = "Retrying $modelName sequential conditioning positive on the next backend for",
            ) { model, _, acquire ->
                phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.GENERATING, acquire.backend.name)
                model.precomputeCondition(params.prompt, "", params.width, params.height, -1)
            }
            val uncondChroma = requestExecutor.withRuntimeModel(
                spec = plan.conditioningRequest.spec,
                options = plan.conditioningRequest.options,
                retryMessage = "Retrying $modelName sequential conditioning negative on the next backend for",
            ) { model, _, _ ->
                model.precomputeCondition(params.negative, "", params.width, params.height, -1)
            }

            requestExecutor.invalidateRuntime(
                plan.conditioningRequest.spec,
                plan.conditioningRequest.options,
            )
            AndroidLogAdapter.i(logTag, "$modelName sequential: Chroma encoder runtime invalidated before diffusion phase")

            cond = condChroma
            uncond = uncondChroma
        } else {
            val condAndUncond = requestExecutor.withRuntimeModel(
                spec = plan.conditioningRequest.spec,
                options = plan.conditioningRequest.options,
                retryMessage = "Retrying $modelName sequential conditioning on the next backend for",
            ) { model, _, acquire ->
                phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.GENERATING, acquire.backend.name)
                val c = model.precomputeCondition(params.prompt, "", params.width, params.height, -1)
                c to null
            }
            cond = condAndUncond.first
            uncond = condAndUncond.second

            requestExecutor.invalidateRuntime(
                plan.conditioningRequest.spec,
                plan.conditioningRequest.options,
            )
            AndroidLogAdapter.i(logTag, "$modelName sequential: encoder runtime invalidated before diffusion phase")
        }

        return requestExecutor.withRuntimeModel(
            spec = plan.diffusionRequest.spec,
            options = plan.diffusionRequest.options,
            retryMessage = "Retrying $modelName sequential diffusion on the next backend for",
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
                    negative = if (isSd3 || isChroma) params.negative else "",
                    width = params.width,
                    height = params.height,
                    steps = params.steps,
                    cfg = params.cfgScale,
                    seed = params.seed,
                    cond = cond,
                    uncond = uncond,
                ) ?: throw IllegalStateException("$modelName sequential generation returned null")
            rgbToBitmap(rgb, params.width, params.height)
        }
    }

    private fun combineSD3Condition(a: PrecomputedCondition, b: PrecomputedCondition): PrecomputedCondition {
        val crossAttn = combineArrays(a.cCrossAttn, a.cCrossAttnDims, b.cCrossAttn, b.cCrossAttnDims, "cCrossAttn", true)
        val vector = combineArrays(a.cVector, a.cVectorDims, b.cVector, b.cVectorDims, "cVector", true)
        val concat = combineArrays(a.cConcat, a.cConcatDims, b.cConcat, b.cConcatDims, "cConcat", false)
        return PrecomputedCondition(
            cCrossAttn = crossAttn.first,
            cCrossAttnDims = crossAttn.second,
            cVector = vector.first,
            cVectorDims = vector.second,
            cConcat = concat.first,
            cConcatDims = concat.second
        )
    }

    private fun combineArrays(
        arrA: FloatArray?, dimsA: IntArray?,
        arrB: FloatArray?, dimsB: IntArray?,
        fieldName: String,
        requiredForSd3: Boolean
    ): Pair<FloatArray?, IntArray?> {
        if (arrA == null && arrB == null) {
            if (requiredForSd3) {
                throw IllegalArgumentException("SD3 conditioning requires field $fieldName but it was null on both sides")
            }
            return null to null
        }
        if (arrA == null) {
            if (requiredForSd3) {
                throw IllegalArgumentException("SD3 conditioning requires field $fieldName on both sides but it was null on side A")
            }
            return arrB!!.clone() to dimsB?.clone()
        }
        if (arrB == null) {
            if (requiredForSd3) {
                throw IllegalArgumentException("SD3 conditioning requires field $fieldName on both sides but it was null on side B")
            }
            return arrA!!.clone() to dimsA?.clone()
        }
        if (dimsA == null || dimsB == null) {
            throw IllegalArgumentException("SD3 conditioning $fieldName is missing dimensions on one side")
        }
        val cleanA = removeTrailingSingletons(dimsA)
        val cleanB = removeTrailingSingletons(dimsB)
        if (!cleanA.contentEquals(cleanB)) {
            throw IllegalArgumentException("SD3 conditioning $fieldName dimensions mismatch: ${dimsA.joinToString()} vs ${dimsB.joinToString()}")
        }
        val sizeA = arrA.size
        val sizeB = arrB.size
        if (sizeA != sizeB) {
            throw IllegalArgumentException("SD3 conditioning $fieldName size mismatch: $sizeA vs $sizeB")
        }
        val result = FloatArray(sizeA) { i -> arrA[i] + arrB[i] }
        val chosenDims = if (dimsB.size > dimsA.size) dimsB else dimsA
        return result to chosenDims.clone()
    }

    private fun removeTrailingSingletons(dims: IntArray): IntArray {
        var lastIdx = dims.size - 1
        while (lastIdx >= 0 && dims[lastIdx] == 1) {
            lastIdx--
        }
        return dims.copyOfRange(0, lastIdx + 1)
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

    private suspend fun generateDirect(
        params: ImageGenerationRequest,
        request: PlannedDiffusionRuntimeRequest,
    ): Bitmap {
        state.resetForRequest()
        val requestId = imageRequestIds.incrementAndGet()
        AndroidLogAdapter.i(
            logTag,
            "Image request entering runtime acquisition: requestId=$requestId size=${params.width}x${params.height} steps=${params.steps}",
        )
        return requestExecutor.withRuntimeModel(
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
