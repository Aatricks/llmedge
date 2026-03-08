package io.aatricks.llmedge

import android.graphics.Bitmap
import android.os.Debug
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.vision.ImageUtils
import kotlin.math.min

internal object StableDiffusionOutputSupport {
    private const val MIN_FRAME_BATCH = 4
    private const val MAX_FRAME_BATCH = 8

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

    fun recoverPotentiallyBlackFrames(
        logTag: String,
        frameBytes: Array<ByteArray>,
    ): Array<ByteArray> {
        val avg = frameBytes.map { computeAverageBrightness(it) }.average()
        AndroidLogAdapter.d(
            logTag,
            "Video frame analysis: ${frameBytes.size} frames, avg brightness=$avg, first frame size=${frameBytes.firstOrNull()?.size ?: 0}",
        )
        if (avg >= 1.0) {
            return frameBytes
        }

        AndroidLogAdapter.w(
            logTag,
            "Detected potentially black frames (avg brightness < 1.0), attempting channel swap...",
        )
        val swapped =
            frameBytes
                .map { bytes ->
                    val out = ByteArray(bytes.size)
                    var j = 0
                    var k = 0
                    while (k + 2 < bytes.size) {
                        val r = bytes[k]
                        val g = bytes[k + 1]
                        val b = bytes[k + 2]
                        out[j++] = b
                        out[j++] = g
                        out[j++] = r
                        k += 3
                    }
                    out
                }
                .toTypedArray()
        val swappedAvg = swapped.map { computeAverageBrightness(it) }.average()
        AndroidLogAdapter.d(logTag, "After BGR swap: avg brightness=$swappedAvg")
        if (swappedAvg > avg) {
            AndroidLogAdapter.w(logTag, "Swapped RGB->BGR for video frames to recover non-black output")
            return swapped
        }

        if (avg < 0.1) {
            frameBytes.firstOrNull()?.let { firstFrame ->
                if (firstFrame.size >= 30) {
                    val sampleBytes = firstFrame.take(30).map { it.toInt() and 0xFF }
                    AndroidLogAdapter.e(
                        logTag,
                        "Frame appears completely black. First 30 bytes: $sampleBytes",
                    )
                }
            }
        }
        return frameBytes
    }

    fun convertFramesToBitmaps(
        frameBytes: Array<ByteArray>,
        width: Int,
        height: Int,
        onRemainingFrames: ((remaining: Int) -> Unit)? = null,
    ): List<Bitmap> {
        val batchSize = determineBatchSize(frameBytes.size)
        val bitmaps = ArrayList<Bitmap>(frameBytes.size)
        val pixelBuffer = IntArray(width * height)
        var index = 0
        while (index < frameBytes.size) {
            val end = min(index + batchSize, frameBytes.size)
            for (i in index until end) {
                val bytesCopy = frameBytes[i].clone()
                bitmaps += ImageUtils.rgbBytesToBitmap(bytesCopy, width, height, pixelBuffer)
            }
            val remaining = frameBytes.size - end
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

    private fun computeAverageBrightness(bytes: ByteArray): Double {
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