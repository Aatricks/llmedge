package io.aatricks.llmedge.image.diffusion

import android.graphics.Bitmap
import android.os.Debug
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.vision.ImageUtils
import kotlin.math.min

internal object StableDiffusionOutputSupport {
    private const val MIN_FRAME_BATCH = 4
    private const val MAX_FRAME_BATCH = 8

    private fun normalizeFrameToRgb24(
        logTag: String,
        bytes: ByteArray,
        width: Int,
        height: Int,
    ): ByteArray {
        val pixelCount = (width * height).coerceAtLeast(1)
        val expectedRgb = pixelCount * 3
        val expectedRgba = pixelCount * 4

        return when (bytes.size) {
            expectedRgb -> bytes
            expectedRgba -> {
                AndroidLogAdapter.w(logTag, "Frame appears to be RGBA/BGRA (${bytes.size} bytes). Stripping alpha to RGB24.")
                val out = ByteArray(expectedRgb)
                var src = 0
                var dst = 0
                while (src + 3 < bytes.size && dst + 2 < out.size) {
                    out[dst++] = bytes[src]
                    out[dst++] = bytes[src + 1]
                    out[dst++] = bytes[src + 2]
                    src += 4
                }
                out
            }
            else -> {
                throw IllegalArgumentException(
                    "Unexpected frame byte size=${bytes.size}. Expected $expectedRgb (RGB24) or $expectedRgba (RGBA32) for ${width}x${height}.",
                )
            }
        }
    }

    fun normalizeFramesToRgb24(
        logTag: String,
        frameBytes: Array<ByteArray>,
        width: Int,
        height: Int,
    ): Array<ByteArray> {
        if (frameBytes.isEmpty()) return frameBytes
        return frameBytes.map { normalizeFrameToRgb24(logTag, it, width, height) }.toTypedArray()
    }

    fun bitmapToRgbBytes(
        bitmap: Bitmap,
        reusableBuffer: ThreadLocal<ByteArray>,
    ): Triple<ByteArray, Int, Int> {
        val safeBitmap =
            if (bitmap.config == Bitmap.Config.ARGB_8888) {
                bitmap
            } else {
                bitmap.copy(Bitmap.Config.ARGB_8888, false)
            }
        val width = safeBitmap.width
        val height = safeBitmap.height
        val pixels = IntArray(width * height)
        safeBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val requiredSize = width * height * 3
        var rgb = reusableBuffer.get()
        if (rgb == null || rgb.size < requiredSize) {
            rgb = ByteArray(requiredSize)
            reusableBuffer.set(rgb)
        }
        var rgbIndex = 0
        for (pixel in pixels) {
            rgb[rgbIndex++] = ((pixel shr 16) and 0xFF).toByte()
            rgb[rgbIndex++] = ((pixel shr 8) and 0xFF).toByte()
            rgb[rgbIndex++] = (pixel and 0xFF).toByte()
        }
        return Triple(rgb, width, height)
    }

    fun logVideoFrameStats(
        logTag: String,
        frameBytesRgb24: Array<ByteArray>,
    ) {
        if (frameBytesRgb24.isEmpty()) return
        val avg = frameBytesRgb24.map { computeAverageBrightnessRgb24(it) }.average()
        AndroidLogAdapter.d(
            logTag,
            "Video frame analysis: ${frameBytesRgb24.size} frames, avg brightness=$avg, first frame size=${frameBytesRgb24.firstOrNull()?.size ?: 0}",
        )

        if (avg < 0.1) {
            val first = frameBytesRgb24.first()
            val sample = first.take(30).map { it.toInt() and 0xFF }
            val nonZero = first.count { it.toInt() != 0 }
            AndroidLogAdapter.e(
                logTag,
                "Frames appear nearly black (avg brightness < 0.1). first30=$sample nonZeroBytes=$nonZero/${first.size}",
            )
        }
    }

    fun convertFramesToBitmaps(
        frameBytesRgb24: Array<ByteArray>,
        width: Int,
        height: Int,
        onRemainingFrames: ((remaining: Int) -> Unit)? = null,
    ): List<Bitmap> {
        val batchSize = determineBatchSize(frameBytesRgb24.size)
        val bitmaps = ArrayList<Bitmap>(frameBytesRgb24.size)
        val pixelBuffer = IntArray(width * height)
        var index = 0
        while (index < frameBytesRgb24.size) {
            val end = min(index + batchSize, frameBytesRgb24.size)
            for (i in index until end) {
                bitmaps += ImageUtils.rgbBytesToBitmap(frameBytesRgb24[i], width, height, pixelBuffer)
            }
            val remaining = frameBytesRgb24.size - end
            if (remaining > 0) {
                onRemainingFrames?.invoke(remaining)
            }
            index = end
        }
        return bitmaps
    }

    fun estimateFrameFootprintBytes(width: Int, height: Int, frameCount: Int): Long {
        val pixels = width.toLong() * height.toLong()
        return pixels * 4L * frameCount
    }

    fun warnIfLowMemory(
        logTag: String,
        estimatedAdditionalBytes: Long,
        bytesInMb: Long,
        memoryPressureThreshold: Float,
    ) {
        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory().coerceAtLeast(bytesInMb)
        val used = runtime.totalMemory() - runtime.freeMemory()
        val projected = used + estimatedAdditionalBytes.coerceAtLeast(0L)
        val ratio = projected.toDouble() / maxMemory.toDouble()
        if (ratio >= memoryPressureThreshold) {
            AndroidLogAdapter.w(
                logTag,
                "Memory pressure warning: projected ${(projected / bytesInMb)} MB of ${(maxMemory / bytesInMb)} MB heap",
            )
        }
    }

    fun readNativeMemoryMb(bytesInMb: Long): Long =
        try {
            Debug.getNativeHeapAllocatedSize().coerceAtLeast(0L) / bytesInMb
        } catch (_: Throwable) {
            val runtime = Runtime.getRuntime()
            (runtime.totalMemory() - runtime.freeMemory()) / bytesInMb
        }

    private fun computeAverageBrightnessRgb24(bytes: ByteArray): Double {
        var sum = 0L
        var index = 0
        val totalPixels = (bytes.size / 3).coerceAtLeast(1)
        while (index + 2 < bytes.size) {
            val r = bytes[index++].toInt() and 0xFF
            val g = bytes[index++].toInt() and 0xFF
            val b = bytes[index++].toInt() and 0xFF
            sum += (r + g + b) / 3
        }
        return sum.toDouble() / totalPixels
    }

    private fun determineBatchSize(frameCount: Int): Int =
        when {
            frameCount >= 48 -> MIN_FRAME_BATCH
            frameCount >= 24 -> 6
            else -> MAX_FRAME_BATCH
        }
}