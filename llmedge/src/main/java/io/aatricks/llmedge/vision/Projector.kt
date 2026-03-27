package io.aatricks.llmedge.vision

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.NativeLibraryLoader

/**
 * Holds vision embeddings produced by native projector encoding, keeping all data in memory
 * so no temporary files are needed.
 */
data class VisionEmbeddings(
    val data: FloatArray,
    val nTokens: Int,
    val nx: Int,
    val ny: Int,
    val embdDim: Int,
    val useMrope: Boolean,
    val useNonCausal: Boolean,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as VisionEmbeddings
        return data.contentEquals(other.data) &&
            nTokens == other.nTokens && nx == other.nx && ny == other.ny &&
            embdDim == other.embdDim && useMrope == other.useMrope &&
            useNonCausal == other.useNonCausal
    }

    override fun hashCode(): Int {
        var result = data.contentHashCode()
        result = 31 * result + nTokens
        result = 31 * result + nx
        result = 31 * result + ny
        result = 31 * result + embdDim
        result = 31 * result + useMrope.hashCode()
        result = 31 * result + useNonCausal.hashCode()
        return result
    }
}

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
    private external fun nativeEncodeImageBuffer(nativePtr: Long, jpegData: ByteArray): Array<Any>?
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

    internal fun nativeHandle(): Long = nativePtr

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

    /**
     * Encode a JPEG image from an in-memory byte array and return embeddings directly,
     * avoiding all temporary file I/O.
     */
    fun encodeImageBuffer(jpegData: ByteArray): VisionEmbeddings? {
        return try {
            if (nativePtr == 0L) {
                AndroidLogAdapter.w(TAG, "Projector native support unavailable; cannot encode image buffer")
                null
            } else {
                val result = nativeEncodeImageBuffer(nativePtr, jpegData) ?: return null
                val embdData = result[0] as FloatArray
                val meta = result[1] as IntArray
                VisionEmbeddings(
                    data = embdData,
                    nTokens = meta[0],
                    nx = meta[1],
                    ny = meta[2],
                    embdDim = meta[3],
                    useMrope = meta[4] != 0,
                    useNonCausal = meta[5] != 0,
                )
            }
        } catch (e: UnsatisfiedLinkError) {
            AndroidLogAdapter.w(TAG, "nativeEncodeImageBuffer not available: ${e.message}")
            null
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
