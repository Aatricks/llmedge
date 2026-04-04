package io.aatricks.llmedge.image

import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.runtime.CpuTopology

internal data class PlannedDiffusionRuntimeRequest(
    val spec: DiffusionRuntimeSpec,
    val options: DiffusionLoadOptions,
)

internal object ImageRuntimeRequestPlanner {
    fun imageRequest(
        params: ImageGenerationRequest,
        config: LLMEdgeConfig,
    ): PlannedDiffusionRuntimeRequest =
        PlannedDiffusionRuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.IMAGE,
                    model = params.model ?: config.models.image,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.IMAGE,
                    // Keep GPU backends eligible by default. preferPerformanceMode tunes
                    // heuristics, but should not silently force CPU or CPU-offloaded weights.
                    allowGpu = true,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = false,
                    keepClipOnCpu = false,
                    keepVaeOnCpu = false,
                    flashAttn = params.flashAttention,
                    vaeDecodeOnly = true,
                    // Image generation should stay on the direct path unless the caller
                    // explicitly forces sequential loading. The generic diffusion heuristic
                    // is tuned for larger video/text-encoder loads and can incorrectly
                    // downgrade GPU-capable image generation into CPU-heavy mode.
                    sequentialLoad = params.forceSequentialLoad,
                    preferPerformanceMode = config.image.preferPerformanceMode,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                ),
        )

    fun directVideoRequest(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): PlannedDiffusionRuntimeRequest {
        val usingCustomTae = params.taehv != null
        return PlannedDiffusionRuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.VIDEO,
                    model = params.model ?: config.models.video.diffusion,
                    vae = if (usingCustomTae) null else (params.vae ?: config.models.video.vae),
                    textEncoder = params.textEncoder ?: config.models.video.textEncoder,
                    taehv = params.taehv,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.VIDEO,
                    allowGpu = !usingCustomTae,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = usingCustomTae || !config.image.preferPerformanceMode,
                    keepClipOnCpu = usingCustomTae || !config.image.preferPerformanceMode,
                    keepVaeOnCpu = usingCustomTae || !config.image.preferPerformanceMode,
                    flashAttn = params.flashAttention,
                    vaeDecodeOnly = params.initImage == null,
                    preferPerformanceMode = config.image.preferPerformanceMode,
                    flowShift = params.flowShift,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                ),
        )
    }

    fun sequentialVideoConditioningRequest(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): PlannedDiffusionRuntimeRequest {
        val usingCustomTae = params.taehv != null
        return PlannedDiffusionRuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.VIDEO_TEXT_ENCODER,
                    model = params.textEncoder ?: config.models.video.textEncoder,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.VIDEO,
                    allowGpu = !usingCustomTae,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
                    offloadToCpu = true,
                    keepClipOnCpu = true,
                    keepVaeOnCpu = true,
                    flashAttn = params.flashAttention,
                    preferPerformanceMode = config.image.preferPerformanceMode,
                ),
        )
    }

    fun sequentialVideoDiffusionRequest(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): PlannedDiffusionRuntimeRequest {
        val usingCustomTae = params.taehv != null
        return PlannedDiffusionRuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.VIDEO,
                    model = params.model ?: config.models.video.diffusion,
                    vae = if (usingCustomTae) null else (params.vae ?: config.models.video.vae),
                    taehv = params.taehv,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.VIDEO,
                    allowGpu = !usingCustomTae,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    offloadToCpu = true,
                    keepClipOnCpu = true,
                    keepVaeOnCpu = true,
                    flashAttn = params.flashAttention,
                    vaeDecodeOnly = params.initImage == null,
                    preferPerformanceMode = config.image.preferPerformanceMode,
                    flowShift = params.flowShift,
                    loraModelDir = params.loraModelDir,
                    loraApplyMode = params.loraApplyMode,
                ),
        )
    }
}
