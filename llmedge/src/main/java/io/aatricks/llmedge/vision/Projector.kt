package io.aatricks.llmedge.vision

import android.util.Log
import io.aatricks.llmedge.core.NativeLibraryLoader

/**
 * Native-backed helper for preparing images using an mmproj file.
 *
 * When the native projector is available this class calls into JNI to
 * produce prepared embeddings. When the native library is not present it
 * falls back to copying the input image to the output path so callers can
 * continue to operate in a degraded mode.
 */
class Projector : AutoCloseable {
    companion object {
        private const val TAG = "Projector"

        init {
            NativeLibraryLoader.ensureSmolLMLoaded(
                required = false,
                onDebug = { message -> Log.d(TAG, message) },
                onError = { message, throwable -> Log.d(TAG, "$message: ${throwable?.message}") },
            )
        }
    }

    private var nativePtr: Long = 0L

    private external fun nativeInitProjector(mmprojPath: String, textModelPtr: Long): Long
    private external fun nativeEncodeImage(nativePtr: Long, imagePath: String, outPath: String): Boolean
    private external fun nativeCloseProjector(nativePtr: Long)

    /** Initialize projector without a native text model pointer. */
    fun init(mmprojPath: String) {
        nativePtr = try {
            nativeInitProjector(mmprojPath, 0L)
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "nativeInitProjector not available: ${e.message}")
            0L
        }
    }

    /**
     * Initialize projector with an optional native text model pointer. The
     * pointer is only used for native-side validation and must not be freed by
     * the caller.
     */
    fun init(mmprojPath: String, textModelPtr: Long) {
        nativePtr = try {
            nativeInitProjector(mmprojPath, textModelPtr)
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "nativeInitProjector not available: ${e.message}")
            0L
        }
    }

    fun encodeImageToFile(imagePath: String, outPath: String): Boolean {
        return try {
            if (nativePtr == 0L) {
                // If native not available, just copy the file as a best-effort placeholder
                val src = java.io.File(imagePath)
                val dst = java.io.File(outPath)
                src.copyTo(dst, overwrite = true)
                true
            } else {
                nativeEncodeImage(nativePtr, imagePath, outPath)
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "nativeEncodeImage not available: ${e.message}")
            try {
                val src = java.io.File(imagePath)
                val dst = java.io.File(outPath)
                src.copyTo(dst, overwrite = true)
                true
            } catch (ex: Exception) {
                Log.e(TAG, "fallback copy failed: ${ex.message}")
                false
            }
        }
    }

    override fun close() {
        try {
            if (nativePtr != 0L) {
                nativeCloseProjector(nativePtr)
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "nativeCloseProjector not available: ${e.message}")
        } finally {
            nativePtr = 0L
        }
    }
}
