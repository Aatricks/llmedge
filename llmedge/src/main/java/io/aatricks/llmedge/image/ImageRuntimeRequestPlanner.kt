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
    ): PlannedDiffusionRuntimeRequest =
        PlannedDiffusionRuntimeRequest(
            spec =
                baseVideoSpec(
                    params = params,
                    config = config,
                    role = DiffusionRuntimeRole.VIDEO,
                    includeTextEncoder = true,
                ),
            options =
                baseVideoOptions(
                    params = params,
                    config = config,
                    taskType = CpuTopology.TaskType.DIFFUSION,
                    offloadAllToCpu = usesCpuOnlyVideoPath(params, config),
                    vaeDecodeOnly = params.initImage == null,
                ),
        )

    fun sequentialVideoConditioningRequest(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): PlannedDiffusionRuntimeRequest =
        PlannedDiffusionRuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.VIDEO_TEXT_ENCODER,
                    model = params.textEncoder ?: config.models.video.textEncoder,
                ),
            options =
                baseVideoOptions(
                    params = params,
                    config = config,
                    taskType = CpuTopology.TaskType.PROMPT_PROCESSING,
                    offloadAllToCpu = true,
                    vaeDecodeOnly = true,
                ),
        )

    fun sequentialVideoDiffusionRequest(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): PlannedDiffusionRuntimeRequest =
        PlannedDiffusionRuntimeRequest(
            spec =
                baseVideoSpec(
                    params = params,
                    config = config,
                    role = DiffusionRuntimeRole.VIDEO,
                    includeTextEncoder = false,
                ),
            options =
                baseVideoOptions(
                    params = params,
                    config = config,
                    taskType = CpuTopology.TaskType.DIFFUSION,
                    offloadAllToCpu = true,
                    vaeDecodeOnly = params.initImage == null,
                ),
        )

    private fun baseVideoSpec(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
        role: DiffusionRuntimeRole,
        includeTextEncoder: Boolean,
    ): DiffusionRuntimeSpec {
        val usingCustomTae = params.taehv != null
        return DiffusionRuntimeSpec(
            role = role,
            model = params.model ?: config.models.video.diffusion,
            vae = if (usingCustomTae) null else (params.vae ?: config.models.video.vae),
            textEncoder = if (includeTextEncoder) params.textEncoder ?: config.models.video.textEncoder else null,
            taehv = params.taehv,
        )
    }

    private fun baseVideoOptions(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
        taskType: CpuTopology.TaskType,
        offloadAllToCpu: Boolean,
        vaeDecodeOnly: Boolean,
    ): DiffusionLoadOptions {
        val allowGpu = params.taehv == null
        return DiffusionLoadOptions(
            subsystem = ComputeSubsystem.VIDEO,
            allowGpu = allowGpu,
            nThreads = CpuTopology.getOptimalThreadCount(taskType),
            offloadToCpu = offloadAllToCpu,
            keepClipOnCpu = offloadAllToCpu,
            keepVaeOnCpu = offloadAllToCpu,
            flashAttn = params.flashAttention,
            vaeDecodeOnly = vaeDecodeOnly,
            preferPerformanceMode = config.image.preferPerformanceMode,
            flowShift = params.flowShift,
            loraModelDir = params.loraModelDir,
            loraApplyMode = params.loraApplyMode,
        )
    }

    private fun usesCpuOnlyVideoPath(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): Boolean = params.taehv != null || !config.image.preferPerformanceMode
}
