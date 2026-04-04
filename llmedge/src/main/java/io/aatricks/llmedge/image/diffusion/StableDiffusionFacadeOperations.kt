package io.aatricks.llmedge.image.diffusion

import android.graphics.Bitmap
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.image.diffusion.internal.StableDiffusionExecutor

internal object StableDiffusionFacadeOperations {
    fun isVideoModel(instance: StableDiffusion): Boolean {
        val metadata = instance.state.modelMetadata ?: return false
        return StableDiffusionMetadataSupport.isVideoModel(metadata)
    }

    suspend fun txt2vid(
        instance: StableDiffusion,
        params: VideoGenerateParams,
        onProgress: VideoProgressCallback? = null,
    ): List<Bitmap> = StableDiffusionExecutor.txt2vid(instance, params, onProgress)

    fun setProgressCallback(instance: StableDiffusion, callback: VideoProgressCallback?) {
        StableDiffusionExecutor.setProgressCallback(instance, callback)
    }

    fun cancelGeneration(instance: StableDiffusion) {
        StableDiffusionExecutor.cancelGeneration(instance)
    }

    fun beginImageRequestTrace(instance: StableDiffusion, requestId: Long) {
        instance.state.beginImageTrace(requestId)
    }

    fun traceImagePhase(
        instance: StableDiffusion,
        phase: ImageGenerationPhase,
        detail: String? = null,
        throwable: Throwable? = null,
    ) {
        if (phase.isTerminal() && instance.state.currentImagePhase?.isTerminal() == true) {
            return
        }
        val requestId = instance.state.appendImageTrace(phase, detail) ?: return
        val message =
            buildString {
                append("requestId=")
                append(requestId)
                append(", phase=")
                append(phase.name)
                if (!detail.isNullOrBlank()) {
                    append(", detail=")
                    append(detail)
                }
            }
        if (throwable == null) {
            AndroidLogAdapter.i("StableDiffusionTrace", message)
        } else {
            AndroidLogAdapter.e("StableDiffusionTrace", message, throwable)
        }
    }

    fun clearImageRequestTrace(instance: StableDiffusion) {
        instance.state.clearImageTraceState()
    }

    fun getLastImageRequestTraceForTests(instance: StableDiffusion): List<ImageGenerationTraceEvent> =
        instance.state.snapshotLastImageTrace()

    fun bitmapToRgbBytesForExecution(
        instance: StableDiffusion,
        bitmap: Bitmap,
    ): Triple<ByteArray, Int, Int> = instance.bitmapToRgbBytes(bitmap)

    fun convertFramesToBitmapsForExecution(
        instance: StableDiffusion,
        frameBytesRgb24: Array<ByteArray>,
        width: Int,
        height: Int,
    ): List<Bitmap> = instance.convertFramesToBitmaps(frameBytesRgb24, width, height)

    fun warnIfLowMemoryForExecution(instance: StableDiffusion, estimatedAdditionalBytes: Long) {
        instance.warnIfLowMemory(estimatedAdditionalBytes)
    }

    fun estimateFrameFootprintBytesForExecution(
        instance: StableDiffusion,
        width: Int,
        height: Int,
        frameCount: Int,
    ): Long = instance.estimateFrameFootprintBytes(width, height, frameCount)

    fun readNativeMemoryMbForExecution(instance: StableDiffusion): Long = instance.readNativeMemoryMb()

    fun nativeIsEasyCacheSupportedForExecution(instance: StableDiffusion): Boolean =
        instance.nativeIsEasyCacheSupported(instance.handleForExecution())

    suspend fun txt2img(instance: StableDiffusion, params: GenerateParams): Bitmap {
        traceImagePhase(
            instance,
            ImageGenerationPhase.TXT2IMG_ENTER,
            "StableDiffusion.txt2img entered width=${params.width} height=${params.height} steps=${params.steps}",
        )
        return StableDiffusionExecutor.txt2img(instance, params)
    }

    fun isEasyCacheSupported(instance: StableDiffusion): Boolean =
        StableDiffusionExecutor.isEasyCacheSupported(instance)
}
