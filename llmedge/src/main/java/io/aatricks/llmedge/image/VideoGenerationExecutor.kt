package io.aatricks.llmedge.image

import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

internal class VideoGenerationExecutor(
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val generationMutex: Mutex,
    private val state: ImageClientState,
    private val requestExecutor: DiffusionRequestExecutor,
) {
    fun generate(params: VideoGenerationRequest): Flow<GenerationStreamEvent> =
        callbackFlow {
            val job =
                scope.coroutineScope.launch {
                    try {
                        val frames =
                            generationMutex.withLock {
                                state.lastGenerationMetrics = null
                                if (params.forceSequentialLoad) {
                                    generateSequentially(params) { message, current, total ->
                                        trySend(
                                            GenerationStreamEvent.Progress(
                                                ProgressEvent.Step(message, current, total),
                                            ),
                                        )
                                    }
                                } else {
                                    generateDirect(params) { message, current, total ->
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
                state.activeModel?.cancelGeneration()
            }
        }

    private suspend fun generateDirect(
        params: VideoGenerationRequest,
        onProgress: ((String, Int, Int) -> Unit)? = null,
    ): List<Bitmap> {
        val request = ImageRuntimeRequestPlanner.directVideoRequest(params, config)
        return requestExecutor.executeWithRuntimeRetry(
            spec = request.spec,
            options = request.options,
            retryMessage = "Retrying video generation on the next backend after a backend-specific failure for",
        ) { acquired ->
            acquired.runtime.mutex.withLock {
                requestExecutor.withActiveModel(acquired.runtime.model) { model ->
                    val easyCache = resolveEasyCache(model, params.easyCache)
                    model.txt2vid(
                        params =
                            toVideoParams(
                                params = params,
                                easyCache = easyCache,
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
                        state.lastGenerationMetrics = model.getLastGenerationMetrics()
                    }
                }
            }
        }
    }

    private suspend fun generateSequentially(
        params: VideoGenerationRequest,
        onProgress: ((String, Int, Int) -> Unit)? = null,
    ): List<Bitmap> {
        val conditioningRequest =
            ImageRuntimeRequestPlanner.sequentialVideoConditioningRequest(params, config)
        val conditioning =
            requestExecutor.executeWithRuntimeRetry(
                spec = conditioningRequest.spec,
                options = conditioningRequest.options,
                retryMessage = "Retrying sequential video text conditioning on the next backend after a backend-specific failure for",
            ) { acquired ->
                acquired.runtime.mutex.withLock {
                    requestExecutor.withActiveModel(acquired.runtime.model) { model ->
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
        return requestExecutor.executeWithRuntimeRetry(
            spec = diffusionRequest.spec,
            options = diffusionRequest.options,
            retryMessage = "Retrying sequential video diffusion on the next backend after a backend-specific failure for",
        ) { acquired ->
            acquired.runtime.mutex.withLock {
                requestExecutor.withActiveModel(acquired.runtime.model) { model ->
                    onProgress?.invoke("Loading diffusion model", 0, params.steps)
                    val easyCache = resolveEasyCache(model, params.easyCache)
                    model.txt2VidWithPrecomputedCondition(
                        params =
                            toVideoParams(
                                params = params,
                                easyCache = easyCache,
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
                        state.lastGenerationMetrics = model.getLastGenerationMetrics()
                    }
                }
            }
        }
    }

    private fun toVideoParams(
        params: VideoGenerationRequest,
        easyCache: EasyCacheParams,
    ): VideoGenerateParams =
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
        )

    private fun resolveEasyCache(
        model: StableDiffusion,
        requested: EasyCacheParams,
    ): EasyCacheParams =
        if (requested.enabled && !model.isEasyCacheSupported()) {
            requested.copy(enabled = false)
        } else {
            requested
        }
}
