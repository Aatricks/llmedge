package io.aatricks.llmedge.image.diffusion.internal

import android.graphics.Bitmap
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.ImageGenerationPhase
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.StableDiffusionMetadataSupport
import io.aatricks.llmedge.image.diffusion.StableDiffusionOutputSupport
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.withLock

internal object StableDiffusionExecutor {
    private const val LOG_TAG = "StableDiffusionExecutor"

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
        instance.updateCachedProgressCallback(callback)
        if (!StableDiffusion.supportIsNativeLibraryAvailable()) {
            return
        }
        instance.bridge.setProgressCallback(instance.state.handle, callback)
    }

    fun cancelGeneration(instance: StableDiffusion) {
        if (instance.state.closed.get()) {
            return
        }
        if (!instance.state.cancellationRequested.compareAndSet(false, true)) {
            return
        }
        if (!StableDiffusion.supportIsNativeLibraryAvailable()) {
            return
        }
        runCatching {
            instance.bridge.cancelGeneration(instance.state.handle)
        }.onFailure { error ->
            if (!instance.state.closed.get()) {
                AndroidLogAdapter.w(
                    LOG_TAG,
                    "cancelGeneration bridge call failed: ${error.message}",
                )
            }
        }
    }

    suspend fun txt2img(instance: StableDiffusion, params: GenerateParams): Bitmap =
        run {
            val dispatchQueuedAtNanos = System.nanoTime()
            withContext(StableDiffusion.diffusionDispatcher) {
                val dispatchDelayMs = ((System.nanoTime() - dispatchQueuedAtNanos) / 1_000_000L).coerceAtLeast(0L)
                instance.traceImagePhase(
                    ImageGenerationPhase.EXECUTOR_ENTER,
                    "dispatchDelayMs=$dispatchDelayMs width=${params.width} height=${params.height} steps=${params.steps}",
                )
                AndroidLogAdapter.i(
                    LOG_TAG,
                    "txt2img executor start dispatchDelayMs=$dispatchDelayMs width=${params.width} height=${params.height} steps=${params.steps}",
                )
                val startNanos = System.nanoTime()
                val memoryBefore = instance.readNativeMemoryMbForExecution()

                try {
                    var argbSupported = true
                    val argbPixels =
                        try {
                            AndroidLogAdapter.i(
                                LOG_TAG,
                                "txt2img attempting native ARGB path",
                            )
                            instance.traceImagePhase(
                                ImageGenerationPhase.WAITING_GENERATION_MUTEX,
                                "about to acquire generation mutex for ARGB path",
                            )
                            instance.state.generationMutex.withLock {
                                instance.state.cancellationRequested.set(false)
                                AndroidLogAdapter.i(
                                    LOG_TAG,
                                    "txt2img acquired generation mutex for native ARGB path",
                                )
                                instance.traceImagePhase(
                                    ImageGenerationPhase.WAITING_GENERATION_MUTEX,
                                    "generation mutex acquired for ARGB path",
                                )
                                instance.traceImagePhase(
                                    ImageGenerationPhase.JNI_ARGB_ENTER,
                                    "calling native ARGB image generation",
                                )
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
                            argbSupported = false
                            AndroidLogAdapter.w(
                                LOG_TAG,
                                "txt2img ARGB JNI binding unavailable; falling back to RGB byte path",
                            )
                            null
                        } catch (t: Throwable) {
                            if (instance.state.cancellationRequested.get()) {
                                instance.state.cancellationRequested.set(false)
                                throw CancellationException("Image generation cancelled", t)
                            }
                            throw t
                        } finally {
                            instance.state.cancellationRequested.set(false)
                        }

                    if (argbPixels != null) {
                        val conversionStart = System.nanoTime()
                        val bitmap =
                            Bitmap.createBitmap(
                                argbPixels,
                                0,
                                params.width,
                                params.width,
                                params.height,
                                Bitmap.Config.ARGB_8888,
                            )
                        val conversionSeconds = (System.nanoTime() - conversionStart) / 1_000_000_000f
                        val totalSeconds = (System.nanoTime() - startNanos) / 1_000_000_000f
                        val memoryAfter = instance.readNativeMemoryMbForExecution()
                        instance.state.lastGenerationMetrics =
                            GenerationMetrics(
                                totalTimeSeconds = totalSeconds,
                                framesPerSecond = if (totalSeconds > 0f) 1f / totalSeconds else 0f,
                                timePerStep = if (params.steps > 0) totalSeconds / params.steps else 0f,
                                peakMemoryUsageMb = maxOf(memoryBefore, memoryAfter),
                                vulkanEnabled = instance.state.vulkanEnabledForMetrics,
                                frameConversionTimeSeconds = conversionSeconds,
                            )
                        instance.traceImagePhase(
                            ImageGenerationPhase.COMPLETED,
                            "ARGB path completed totalMs=${((System.nanoTime() - startNanos) / 1_000_000L).coerceAtLeast(0L)}",
                        )
                        return@withContext bitmap
                    }

                    if (argbSupported) {
                        throw InferenceFailedException(
                            operation = "Stable Diffusion image generation (ARGB path)",
                            detail = "The native runtime reported a generation failure.",
                        )
                    }

                    AndroidLogAdapter.i(
                        LOG_TAG,
                        "txt2img attempting native RGB path",
                    )
                    instance.traceImagePhase(
                        ImageGenerationPhase.WAITING_GENERATION_MUTEX,
                        "about to acquire generation mutex for RGB path",
                    )
                    val bytes =
                        try {
                            instance.state.generationMutex.withLock {
                                instance.state.cancellationRequested.set(false)
                                AndroidLogAdapter.i(
                                    LOG_TAG,
                                    "txt2img acquired generation mutex for native RGB path",
                                )
                                instance.traceImagePhase(
                                    ImageGenerationPhase.WAITING_GENERATION_MUTEX,
                                    "generation mutex acquired for RGB path",
                                )
                                instance.traceImagePhase(
                                    ImageGenerationPhase.JNI_RGB_ENTER,
                                    "calling native RGB image generation",
                                )
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
                        } catch (t: Throwable) {
                            if (instance.state.cancellationRequested.get()) {
                                instance.state.cancellationRequested.set(false)
                                throw CancellationException("Image generation cancelled", t)
                            }
                            throw t
                        } finally {
                            instance.state.cancellationRequested.set(false)
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
                    val conversionStart = System.nanoTime()
                    val bitmap =
                        io.aatricks.llmedge.vision.ImageUtils.rgbBytesToBitmap(
                            bytes,
                            params.width,
                            params.height,
                            pixels,
                        )
                    val conversionSeconds = (System.nanoTime() - conversionStart) / 1_000_000_000f
                    val totalSeconds = (System.nanoTime() - startNanos) / 1_000_000_000f
                    val memoryAfter = instance.readNativeMemoryMbForExecution()
                    instance.state.lastGenerationMetrics =
                        GenerationMetrics(
                            totalTimeSeconds = totalSeconds,
                            framesPerSecond = if (totalSeconds > 0f) 1f / totalSeconds else 0f,
                            timePerStep = if (params.steps > 0) totalSeconds / params.steps else 0f,
                            peakMemoryUsageMb = maxOf(memoryBefore, memoryAfter),
                            vulkanEnabled = instance.state.vulkanEnabledForMetrics,
                            frameConversionTimeSeconds = conversionSeconds,
                        )
                    instance.traceImagePhase(
                        ImageGenerationPhase.COMPLETED,
                        "RGB path completed totalMs=${((System.nanoTime() - startNanos) / 1_000_000L).coerceAtLeast(0L)}",
                    )
                    bitmap
                } catch (cancelled: CancellationException) {
                    instance.state.lastGenerationMetrics = null
                    instance.traceImagePhase(
                        ImageGenerationPhase.CANCELLED,
                        "Image generation cancelled in executor",
                        cancelled,
                    )
                    throw cancelled
                } catch (t: Throwable) {
                    instance.state.lastGenerationMetrics = null
                    instance.traceImagePhase(
                        ImageGenerationPhase.FAILED,
                        "Image generation failed in executor: ${t.message}",
                        t,
                    )
                    throw t
                } finally {
                    instance.state.cancellationRequested.set(false)
                }
            }
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
            instance.state.generationMutex.withLock {
                if (instance.state.closed.get()) {
                    throw IllegalStateException("StableDiffusion is closed")
                }
                instance.bridge.precomputeCondition(
                    instance.state.handle,
                    prompt,
                    negative,
                    width,
                    height,
                    clipSkip,
                    instance.isVideoModel(),
                )
            }
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
            params.validate().getOrThrow()
            check(StableDiffusion.supportIsNativeLibraryAvailable()) {
                "Video generation is unavailable on this platform"
            }
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

            if (params.initImage != null) {
                require(params.initImage.width == params.width && params.initImage.height == params.height) {
                    "Init image dimensions (${params.initImage.width}x${params.initImage.height}) must equal the generation dimensions (${params.width}x${params.height})"
                }
            }

            val (initBytes, initWidth, initHeight) =
                params.initImage?.let { instance.bitmapToRgbBytesForExecution(it) } ?: Triple(null, 0, 0)

            try {
                val startNanos = System.nanoTime()
                val memoryBefore = instance.readNativeMemoryMbForExecution()
                val frameBytes =
                    try {
                        instance.state.generationMutex.withLock {
                            if (instance.state.closed.get()) {
                                throw IllegalStateException("StableDiffusion is closed")
                            }
                            if (onProgress != null) {
                                instance.bridge.setProgressCallback(instance.state.handle, onProgress)
                            }
                            try {
                                instance.state.cancellationRequested.set(false)
                                generateFrames(initBytes, initWidth, initHeight)
                                    ?: throw InferenceFailedException(
                                        operation = "Video generation",
                                        detail = "",
                                    )
                            } finally {
                                if (onProgress != null) {
                                    instance.bridge.setProgressCallback(instance.state.handle, null)
                                }
                            }
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
                        operation = "Video generation",
                        detail = "",
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
