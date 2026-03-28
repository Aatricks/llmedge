package io.aatricks.llmedge.image.diffusion.internal

import android.graphics.Bitmap
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.StableDiffusionMetadataSupport
import io.aatricks.llmedge.image.diffusion.StableDiffusionOutputSupport
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.withLock

internal object StableDiffusionExecutor {
    suspend fun txt2vid(
        instance: StableDiffusion,
        params: VideoGenerateParams,
        onProgress: VideoProgressCallback?,
    ): List<Bitmap> =
        executeVideoGeneration(instance, params, onProgress) { initBytes, initWidth, initHeight ->
            instance.bridge.txt2vid(
                instance.state.handle,
                params.prompt,
                params.negative,
                params.width,
                params.height,
                params.videoFrames,
                params.steps,
                params.cfgScale,
                params.seed,
                params.sampleMethod,
                params.scheduler,
                params.strength,
                initBytes,
                initWidth,
                initHeight,
                params.vaceStrength,
                params.easyCacheParams.enabled,
                params.easyCacheParams.reuseThreshold,
                params.easyCacheParams.startPercent,
                params.easyCacheParams.endPercent,
            )
        }

    fun setProgressCallback(instance: StableDiffusion, callback: VideoProgressCallback?) {
        instance.state.cachedProgressCallback = callback
        if (!StableDiffusion.supportIsNativeLibraryAvailable()) {
            return
        }
        instance.bridge.setProgressCallback(instance.state.handle, callback)
    }

    fun cancelGeneration(instance: StableDiffusion) {
        instance.state.cancellationRequested.set(true)
        if (!StableDiffusion.supportIsNativeLibraryAvailable()) {
            return
        }
        instance.bridge.cancelGeneration(instance.state.handle)
    }

    suspend fun txt2img(instance: StableDiffusion, params: GenerateParams): Bitmap =
        withContext(StableDiffusion.diffusionDispatcher) {
            val argbPixels =
                try {
                    instance.state.generationMutex.withLock {
                        instance.bridge.txt2imgArgb(
                            instance.state.handle,
                            params.prompt,
                            params.negative,
                            params.width,
                            params.height,
                            params.steps,
                            params.cfgScale,
                            params.seed,
                            params.vaeTiling,
                            params.easyCacheParams.enabled,
                            params.easyCacheParams.reuseThreshold,
                            params.easyCacheParams.startPercent,
                            params.easyCacheParams.endPercent,
                        )
                    }
                } catch (_: UnsatisfiedLinkError) {
                    null
                }

            if (argbPixels != null) {
                return@withContext Bitmap.createBitmap(
                    argbPixels,
                    0,
                    params.width,
                    params.width,
                    params.height,
                    Bitmap.Config.ARGB_8888,
                )
            }

            val bytes =
                instance.state.generationMutex.withLock {
                    instance.bridge.txt2img(
                        instance.state.handle,
                        params.prompt,
                        params.negative,
                        params.width,
                        params.height,
                        params.steps,
                        params.cfgScale,
                        params.seed,
                        params.vaeTiling,
                        params.easyCacheParams.enabled,
                        params.easyCacheParams.reuseThreshold,
                        params.easyCacheParams.startPercent,
                        params.easyCacheParams.endPercent,
                    ) ?: throw InferenceFailedException(
                        operation = "Stable Diffusion image generation",
                        detail = "The native runtime reported a generation failure.",
                    )
                }

            val expectedMin = params.width * params.height * 3
            if (bytes.size < expectedMin) {
                StableDiffusion.supportLogWarning(
                    "txt2img returned short RGB buffer: size=${bytes.size}, expectedAtLeast=$expectedMin (w=${params.width}, h=${params.height})",
                )
            }
            val pixelCount = params.width * params.height
            val pixels =
                instance.state.txt2imgPixelBuffer.let { buf ->
                    if (buf != null && buf.size >= pixelCount) {
                        buf
                    } else {
                        IntArray(pixelCount).also { instance.state.txt2imgPixelBuffer = it }
                    }
                }
            io.aatricks.llmedge.vision.ImageUtils.rgbBytesToBitmap(
                bytes,
                params.width,
                params.height,
                pixels,
            )
        }

    fun isEasyCacheSupported(instance: StableDiffusion): Boolean {
        instance.state.easyCacheSupported?.let { return it }

        val supported =
            if (!StableDiffusion.supportIsNativeLibraryAvailable() ||
                StableDiffusion.supportNativeBridgeOverriddenForTests()
            ) {
                instance.state.modelMetadata?.let(StableDiffusionMetadataSupport::supportsEasyCache)
                    ?: false
            } else {
                try {
                    instance.nativeIsEasyCacheSupportedForExecution()
                } catch (_: Throwable) {
                    instance.state.modelMetadata?.let(StableDiffusionMetadataSupport::supportsEasyCache)
                        ?: false
                }
            }

        instance.state.easyCacheSupported = supported
        return supported
    }

    suspend fun precomputeCondition(
        instance: StableDiffusion,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        clipSkip: Int,
    ): PrecomputedCondition? =
        withContext(StableDiffusion.diffusionDispatcher) {
            instance.bridge.precomputeCondition(
                instance.state.handle,
                prompt,
                negative,
                width,
                height,
                clipSkip,
            )
        }

    suspend fun txt2VidWithPrecomputedCondition(
        instance: StableDiffusion,
        params: VideoGenerateParams,
        cond: PrecomputedCondition?,
        uncond: PrecomputedCondition?,
        onProgress: VideoProgressCallback?,
    ): List<Bitmap> =
        executeVideoGeneration(instance, params, onProgress) { initBytes, initWidth, initHeight ->
            instance.bridge.txt2vidWithPrecomputedCondition(
                instance.state.handle,
                params.prompt,
                params.negative,
                params.width,
                params.height,
                params.videoFrames,
                params.steps,
                params.cfgScale,
                params.seed,
                params.sampleMethod,
                params.scheduler,
                params.strength,
                initBytes,
                initWidth,
                initHeight,
                cond,
                uncond,
                params.vaceStrength,
                params.easyCacheParams.enabled,
                params.easyCacheParams.reuseThreshold,
                params.easyCacheParams.startPercent,
                params.easyCacheParams.endPercent,
            )
        }

    private suspend fun executeVideoGeneration(
        instance: StableDiffusion,
        params: VideoGenerateParams,
        onProgress: VideoProgressCallback?,
        generateFrames: (ByteArray?, Int, Int) -> Array<ByteArray>?,
    ): List<Bitmap> =
        withContext(StableDiffusion.diffusionDispatcher) {
            check(StableDiffusion.supportIsNativeLibraryAvailable()) {
                "Video generation is unavailable on this platform"
            }
            params.validate().getOrThrow()
            check(instance.isVideoModel()) { "Loaded model is not a video model (use txt2img instead)" }

            val maxFrames =
                when (instance.state.modelMetadata?.parameterCount) {
                    "5B" -> 32
                    else -> 64
                }
            require(params.videoFrames <= maxFrames) {
                "Model ${instance.state.modelMetadata?.parameterCount ?: "unknown"} supports maximum $maxFrames frames. " +
                    "Requested ${params.videoFrames} frames. Use a smaller model or reduce frame count."
            }

            val estimatedBytes =
                instance.estimateFrameFootprintBytesForExecution(
                    width = params.width,
                    height = params.height,
                    frameCount = params.videoFrames,
                )
            instance.warnIfLowMemoryForExecution(estimatedBytes)

            val (initBytes, initWidth, initHeight) =
                params.initImage?.let { instance.bitmapToRgbBytesForExecution(it) } ?: Triple(null, 0, 0)

            if (onProgress != null) {
                instance.bridge.setProgressCallback(instance.state.handle, onProgress)
            }

            try {
                val startNanos = System.nanoTime()
                val memoryBefore = instance.readNativeMemoryMbForExecution()
                val frameBytes =
                    try {
                        instance.state.generationMutex.withLock {
                            instance.state.cancellationRequested.set(false)
                            generateFrames(initBytes, initWidth, initHeight)
                                ?: throw InferenceFailedException(
                                    operation = "Stable Diffusion video generation",
                                    detail = "The native runtime reported a generation failure.",
                                )
                        }
                    } catch (t: Throwable) {
                        if (instance.state.cancellationRequested.get()) {
                            instance.state.cancellationRequested.set(false)
                            throw CancellationException("Video generation cancelled", t)
                        }
                        throw t
                    } finally {
                        instance.state.cancellationRequested.set(false)
                    }

                if (frameBytes.isEmpty()) {
                    throw InferenceFailedException(
                        operation = "Stable Diffusion video generation",
                        detail = "The native runtime returned no frames.",
                    )
                }

                val expectedFrames = params.actualFrameCount()
                if (frameBytes.size != expectedFrames) {
                    StableDiffusion.supportLogWarning(
                        "Expected $expectedFrames frames (formula: (${params.videoFrames}-1)/4*4+1) but received ${frameBytes.size}",
                    )
                }

                val frameBytesRgb24 =
                    StableDiffusionOutputSupport.normalizeFramesToRgb24(
                        "StableDiffusion",
                        frameBytes,
                        params.width,
                        params.height,
                    )

                StableDiffusionOutputSupport.logVideoFrameStats("StableDiffusion", frameBytesRgb24)

                val conversionStart = System.nanoTime()
                val bitmaps =
                    instance.convertFramesToBitmapsForExecution(
                        frameBytesRgb24,
                        params.width,
                        params.height,
                    )
                val conversionSeconds = (System.nanoTime() - conversionStart) / 1_000_000_000f
                val totalSeconds = (System.nanoTime() - startNanos) / 1_000_000_000f
                val memoryAfter = instance.readNativeMemoryMbForExecution()

                instance.state.lastGenerationMetrics =
                    GenerationMetrics(
                        totalTimeSeconds = totalSeconds,
                        framesPerSecond = if (totalSeconds > 0f) bitmaps.size / totalSeconds else 0f,
                        timePerStep = if (params.steps > 0) totalSeconds / params.steps else 0f,
                        peakMemoryUsageMb = maxOf(memoryBefore, memoryAfter),
                        vulkanEnabled = instance.state.vulkanEnabledForMetrics,
                        frameConversionTimeSeconds = conversionSeconds,
                    )

                instance.warnIfLowMemoryForExecution(estimatedBytes)
                bitmaps
            } finally {
                if (onProgress != null) {
                    instance.bridge.setProgressCallback(
                        instance.state.handle,
                        instance.state.cachedProgressCallback,
                    )
                }
            }
        }
}
