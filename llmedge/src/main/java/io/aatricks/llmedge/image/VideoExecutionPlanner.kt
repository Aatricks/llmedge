package io.aatricks.llmedge.image

import android.app.ActivityManager
import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.InsufficientMemoryException
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import kotlin.math.max

internal data class VideoComponentSizes(
    val diffusionBytes: Long,
    val vaeBytes: Long,
    val textEncoderBytes: Long,
    val highNoiseDiffusionBytes: Long = 0L,
)

internal data class VideoMemorySnapshot(
    val availableSystemBytes: Long,
    val lowMemoryThresholdBytes: Long,
    val totalSystemBytes: Long,
)

internal data class VideoExecutionDecision(
    val mode: ImageExecutionMode,
    val reason: String,
    val directPeakBytes: Long,
    val sequentialPeakBytes: Long,
    val safeBudgetBytes: Long,
)

internal object VideoExecutionPlanner {
    private const val MEBIBYTE = 1024L * 1024L
    private const val MIN_DEVICE_RESERVE_BYTES = 512L * MEBIBYTE
    private const val MIN_WORKSPACE_BYTES = 256L * MEBIBYTE

    fun decide(
        forceSequentialLoad: Boolean,
        sizes: VideoComponentSizes,
        memory: VideoMemorySnapshot,
    ): VideoExecutionDecision {
        val budget = safeBudgetBytes(memory)
        val directPeak =
            estimatePeakBytes(
                sizes.diffusionBytes,
                sizes.vaeBytes,
                sizes.textEncoderBytes,
                sizes.highNoiseDiffusionBytes,
                memory = memory,
            )
        val sequentialPeak =
            max(
                estimatePeakBytes(sizes.textEncoderBytes, memory = memory),
                estimatePeakBytes(
                    sizes.diffusionBytes,
                    sizes.vaeBytes,
                    sizes.highNoiseDiffusionBytes,
                    memory = memory,
                ),
            )
        if (memory.availableSystemBytes <= 0L || memory.totalSystemBytes <= 0L) {
            return VideoExecutionDecision(
                mode = if (forceSequentialLoad) ImageExecutionMode.SEQUENTIAL else ImageExecutionMode.DIRECT,
                reason = if (forceSequentialLoad) "FORCED_SEQUENTIAL" else "AUTO_DIRECT_MEMORY_UNAVAILABLE",
                directPeakBytes = directPeak,
                sequentialPeakBytes = sequentialPeak,
                safeBudgetBytes = budget,
            )
        }
        if (sequentialPeak > budget) {
            throw InsufficientMemoryException(
                requiredBytes = sequentialPeak,
                availableBytes = budget,
                operation = "sequential video loading",
            )
        }
        if (forceSequentialLoad) {
            return VideoExecutionDecision(
                mode = ImageExecutionMode.SEQUENTIAL,
                reason = "FORCED_SEQUENTIAL",
                directPeakBytes = directPeak,
                sequentialPeakBytes = sequentialPeak,
                safeBudgetBytes = budget,
            )
        }
        val directWithSafetyMargin = saturatingMultiply(directPeak, 2L)
        return if (directWithSafetyMargin <= budget) {
            VideoExecutionDecision(
                mode = ImageExecutionMode.DIRECT,
                reason = "AUTO_DIRECT_HEADROOM",
                directPeakBytes = directPeak,
                sequentialPeakBytes = sequentialPeak,
                safeBudgetBytes = budget,
            )
        } else {
            VideoExecutionDecision(
                mode = ImageExecutionMode.SEQUENTIAL,
                reason = "AUTO_SEQUENTIAL_MEMORY",
                directPeakBytes = directPeak,
                sequentialPeakBytes = sequentialPeak,
                safeBudgetBytes = budget,
            )
        }
    }

    private fun safeBudgetBytes(memory: VideoMemorySnapshot): Long {
        val reserve = max(MIN_DEVICE_RESERVE_BYTES, memory.totalSystemBytes.coerceAtLeast(0L) / 8L)
        return (memory.availableSystemBytes - memory.lowMemoryThresholdBytes - reserve).coerceAtLeast(0L)
    }

    private fun estimatePeakBytes(
        vararg components: Long,
        memory: VideoMemorySnapshot,
    ): Long {
        val parameterBytes = components.fold(0L) { total, size -> saturatingAdd(total, size.coerceAtLeast(0L)) }
        val loadOverhead = parameterBytes / 8L
        val workspace = max(MIN_WORKSPACE_BYTES, memory.totalSystemBytes.coerceAtLeast(0L) / 32L)
        return saturatingAdd(saturatingAdd(parameterBytes, loadOverhead), workspace)
    }

    private fun saturatingAdd(left: Long, right: Long): Long =
        if (Long.MAX_VALUE - left < right) Long.MAX_VALUE else left + right

    private fun saturatingMultiply(value: Long, multiplier: Long): Long =
        if (value > Long.MAX_VALUE / multiplier) Long.MAX_VALUE else value * multiplier
}

internal fun interface VideoExecutionPlanSelector {
    suspend fun decide(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): VideoExecutionDecision
}

internal class DefaultVideoExecutionPlanSelector(
    private val context: Context,
    private val resolver: ModelRepository,
    private val memorySnapshotProvider: () -> VideoMemorySnapshot = { captureVideoMemorySnapshot(context) },
    private val fileSizeProvider: (ModelSpec, java.io.File) -> Long = { _, file -> file.length() },
    private val phaseListener: DiffusionPhaseListener? = null,
) : VideoExecutionPlanSelector {
    override suspend fun decide(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): VideoExecutionDecision {
        phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.RESOLVING_MODEL)
        val model = params.model ?: config.models.video.diffusion
        val decoder = params.taehv ?: params.vae ?: config.models.video.vae
        val textEncoder = params.textEncoder ?: config.models.video.textEncoder
        val modelBytes = resolveSize(model)
        val decoderBytes = resolveSize(decoder)
        val textEncoderBytes = resolveSize(textEncoder)
        val highNoiseBytes = params.highNoiseDiffusionModel?.let { resolveSize(it) } ?: 0L
        val memory = memorySnapshotProvider()
        val sizes =
            VideoComponentSizes(
                diffusionBytes = modelBytes,
                vaeBytes = decoderBytes,
                textEncoderBytes = textEncoderBytes,
                highNoiseDiffusionBytes = highNoiseBytes,
            )
        return try {
            VideoExecutionPlanner.decide(params.forceSequentialLoad, sizes, memory).also { decision ->
                logDecision(decision, sizes, memory)
            }
        } catch (error: InsufficientMemoryException) {
            logRefusal(sizes, memory, error)
            throw error
        }
    }

    private suspend fun resolveSize(spec: ModelSpec): Long {
        val file = resolver.resolve(context, spec)
        return fileSizeProvider(spec, file).coerceAtLeast(0L)
    }

    private fun logDecision(
        decision: VideoExecutionDecision,
        sizes: VideoComponentSizes,
        memory: VideoMemorySnapshot,
    ) {
        AndroidLogAdapter.i(
            LOG_TAG,
            "mode=${decision.mode} reason=${decision.reason} sizes=$sizes " +
                "directBytes=${decision.directPeakBytes} sequentialBytes=${decision.sequentialPeakBytes} " +
                "budgetBytes=${decision.safeBudgetBytes} availableBytes=${memory.availableSystemBytes} " +
                "thresholdBytes=${memory.lowMemoryThresholdBytes} totalBytes=${memory.totalSystemBytes}",
        )
    }

    private fun logRefusal(
        sizes: VideoComponentSizes,
        memory: VideoMemorySnapshot,
        error: InsufficientMemoryException,
    ) {
        AndroidLogAdapter.w(
            LOG_TAG,
            "mode=REFUSED reason=SEQUENTIAL_EXCEEDS_BUDGET sizes=$sizes " +
                "requiredBytes=${error.requiredBytes} budgetBytes=${error.availableBytes} " +
                "availableBytes=${memory.availableSystemBytes} thresholdBytes=${memory.lowMemoryThresholdBytes} " +
                "totalBytes=${memory.totalSystemBytes}",
        )
    }

    private companion object {
        const val LOG_TAG = "VideoExecutionPlanner"
    }
}

private fun captureVideoMemorySnapshot(context: Context): VideoMemorySnapshot {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val memoryInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memoryInfo)
    return VideoMemorySnapshot(
        availableSystemBytes = memoryInfo.availMem,
        lowMemoryThresholdBytes = memoryInfo.threshold,
        totalSystemBytes = memoryInfo.totalMem,
    )
}
