package io.aatricks.llmedge.image.ipc

import android.os.Parcelable
import android.os.SharedMemory
import kotlinx.parcelize.Parcelize

/**
 * Parcelable mirrors of the public request/result types for the diffusion worker process.
 * Pixel payloads never cross Binder inline (512x512 ARGB is ~1 MB, at the transaction limit);
 * they ride in ashmem via [IpcFrameBuffer].
 */

@Parcelize
internal data class WorkerInitConfig(
    val cacheMaxEntries: Int,
    val cacheMaxMemoryMb: Long,
    val preferPerformanceMode: Boolean,
    val useVulkan: Boolean,
    /** Entries "SUBSYSTEM:BACKEND" seeded into the worker's in-process backend blacklist. */
    val blacklistSeed: List<String>,
) : Parcelable

@Parcelize
internal data class IpcModelSpec(
    /** "local" or "hf". */
    val type: String,
    val path: String?,
    val repoId: String?,
    val filename: String?,
    val revision: String?,
    val preferredQuantizations: List<String>,
    val token: String?,
    val forceDownload: Boolean,
    val preferSystemDownloader: Boolean,
    // ModelHints, flattened
    val artifactKind: String,
    val capabilities: List<String>,
    val chatTemplate: String?,
    val conversionPrecision: String?,
    val conversionAdapter: String?,
    val conversionTokenizerPre: String?,
) : Parcelable

@Parcelize
internal data class IpcImageRequest(
    val prompt: String,
    val negative: String,
    val width: Int,
    val height: Int,
    val steps: Int,
    val cfgScale: Float,
    val seed: Long,
    val flashAttention: Boolean,
    val forceSequentialLoad: Boolean,
    val easyCacheEnabled: Boolean,
    val easyCacheReuseThreshold: Float,
    val easyCacheStartPercent: Float,
    val easyCacheEndPercent: Float,
    val loraModelDir: String?,
    val loraApplyModeId: Int,
    val model: IpcModelSpec?,
    val vae: IpcModelSpec?,
    val textEncoder: IpcModelSpec?,
    val splitDiffusionModel: Boolean,
    val sequential: Boolean,
) : Parcelable

@Parcelize
internal data class IpcVideoRequest(
    val prompt: String,
    val negative: String,
    val width: Int,
    val height: Int,
    val videoFrames: Int,
    val steps: Int,
    val cfgScale: Float,
    val seed: Long,
    val flowShift: Float,
    val flashAttention: Boolean,
    val forceSequentialLoad: Boolean,
    val initImage: IpcFrameBuffer?,
    val strength: Float,
    val sampleMethodId: Int,
    val schedulerId: Int,
    val easyCacheEnabled: Boolean,
    val easyCacheReuseThreshold: Float,
    val easyCacheStartPercent: Float,
    val easyCacheEndPercent: Float,
    val loraModelDir: String?,
    val loraApplyModeId: Int,
    val taehv: IpcModelSpec?,
    val model: IpcModelSpec?,
    val vae: IpcModelSpec?,
    val textEncoder: IpcModelSpec?,
) : Parcelable

/** ARGB_8888 pixel frames packed contiguously into one ashmem region. */
@Parcelize
internal data class IpcFrameBuffer(
    val memory: SharedMemory,
    val width: Int,
    val height: Int,
    val frameCount: Int,
) : Parcelable

@Parcelize
internal data class IpcGenerationMetrics(
    val totalTimeSeconds: Float,
    val framesPerSecond: Float,
    val timePerStep: Float,
    val peakMemoryUsageMb: Long,
    val vulkanEnabled: Boolean,
    val frameConversionTimeSeconds: Float,
    val hasRequestMetrics: Boolean,
    val runtimeAcquireMs: Long,
    val modelLoadMs: Long,
    val generateMs: Long,
    val cacheHit: Boolean,
    val backend: String,
    val flashAttentionEnabled: Boolean,
    val easyCacheEnabled: Boolean,
    val width: Int,
    val height: Int,
    val steps: Int,
) : Parcelable

@Parcelize
internal data class IpcImageResult(
    val frame: IpcFrameBuffer,
    val metrics: IpcGenerationMetrics?,
) : Parcelable

@Parcelize
internal data class IpcVideoResult(
    val frames: IpcFrameBuffer,
    val metrics: IpcGenerationMetrics?,
) : Parcelable

@Parcelize
internal data class PhaseUpdate(
    /** One of [DiffusionPhases]. */
    val phase: String,
    val backend: String?,
    val step: Int,
    val totalSteps: Int,
    val uptimeMillis: Long,
) : Parcelable

@Parcelize
internal data class IpcFailure(
    val code: Int,
    val exceptionClass: String,
    val message: String?,
    /** Backend the worker was using when it failed, if known. */
    val backend: String?,
) : Parcelable {
    companion object {
        const val CODE_GENERIC = 0
        const val CODE_CANCELLED = 1
        const val CODE_BUSY = 2
    }
}

internal object DiffusionPhases {
    const val REQUESTED = "REQUESTED"
    const val RESOLVING_MODEL = "RESOLVING_MODEL"
    const val LOADING = "LOADING"
    const val GENERATING = "GENERATING"
    const val STEP = "STEP"
    const val COMPLETED = "COMPLETED"
    const val FAILED = "FAILED"
}
