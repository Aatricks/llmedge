package io.aatricks.llmedge.vision

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.NativeLibraryLoader

/**
 * Native-backed helper for preparing images using an mmproj file.
 *
 * This helper does not emulate projector behavior in Kotlin. If native projector support is
 * missing or initialization fails, callers must treat that as an unsupported multimodal setup and
 * fail fast.
 */
class Projector : AutoCloseable {
    companion object {
        private const val TAG = "Projector"

        init {
            NativeLibraryLoader.ensureSmolLMLoaded(
                required = false,
                onDebug = { message -> AndroidLogAdapter.d(TAG, message) },
                onError = { message, throwable -> AndroidLogAdapter.w(TAG, "$message: ${throwable?.message}") },
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
            AndroidLogAdapter.w(TAG, "nativeInitProjector not available: ${e.message}")
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
            AndroidLogAdapter.w(TAG, "nativeInitProjector not available: ${e.message}")
            0L
        }
    }

    fun isReady(): Boolean = nativePtr != 0L

    fun encodeImageToFile(imagePath: String, outPath: String): Boolean {
        return try {
            if (nativePtr == 0L) {
                AndroidLogAdapter.w(TAG, "Projector native support unavailable; cannot prepare image embeddings")
                false
            } else {
                nativeEncodeImage(nativePtr, imagePath, outPath)
            }
        } catch (e: UnsatisfiedLinkError) {
            AndroidLogAdapter.w(TAG, "nativeEncodeImage not available: ${e.message}")
            false
        }
    }

    override fun close() {
        try {
            if (nativePtr != 0L) {
                nativeCloseProjector(nativePtr)
            }
        } catch (e: UnsatisfiedLinkError) {
            AndroidLogAdapter.w(TAG, "nativeCloseProjector not available: ${e.message}")
        } finally {
            nativePtr = 0L
        }
    }
}
