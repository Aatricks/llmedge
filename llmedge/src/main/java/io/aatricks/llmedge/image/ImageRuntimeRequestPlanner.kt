package io.aatricks.llmedge.image

import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.runtime.CpuTopology

internal data class PlannedDiffusionRuntimeRequest(
    val spec: DiffusionRuntimeSpec,
    val options: DiffusionLoadOptions,
)

internal sealed interface DiffusionExecutionPlan {
    data class Direct(
        val request: PlannedDiffusionRuntimeRequest,
    ) : DiffusionExecutionPlan

    data class Sequential(
        val conditioningRequest: PlannedDiffusionRuntimeRequest,
        val diffusionRequest: PlannedDiffusionRuntimeRequest,
    ) : DiffusionExecutionPlan
}

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
                    vae = params.vae,
                    textEncoder = params.textEncoder,
                    diffusionModelOnly = params.diffusionModelOnly,
                    splitDiffusionModel = params.splitDiffusionModel,
                ),
            options =
                DiffusionLoadOptions(
                    subsystem = ComputeSubsystem.IMAGE,
                    // Keep GPU backends eligible by default. preferPerformanceMode tunes
                    // heuristics, but should not silently force CPU or CPU-offloaded weights.
                    allowGpu = config.image.useVulkan,
                    nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                    // Split models (FLUX.2 Klein) bundle a multi-GB DiT + Qwen3 encoder; offload to
                    // CPU and keep the encoder/VAE off-GPU so they fit mobile memory budgets.
                    offloadToCpu = params.splitDiffusionModel,
                    keepClipOnCpu = params.splitDiffusionModel,
                    keepVaeOnCpu = params.splitDiffusionModel,
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

    /**
     * Two-phase plan for FLUX.2 sequential low-memory generation: phase 1 loads ONLY the Qwen3
     * encoder (to precompute the conditioning), phase 2 loads ONLY the DiT (+VAE). Peak RAM is the
     * larger of the two phases, not their sum.
     */
    fun imageSequentialPlan(
        params: ImageGenerationRequest,
        config: LLMEdgeConfig,
    ): DiffusionExecutionPlan.Sequential {
        val encoderSpec = params.textEncoder ?: error("sequential image generation requires a textEncoder")
        val ditModel = params.model ?: config.models.image
        val baseOptions =
            DiffusionLoadOptions(
                subsystem = ComputeSubsystem.IMAGE,
                allowGpu = config.image.useVulkan,
                nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = params.flashAttention,
                vaeDecodeOnly = true,
                sequentialLoad = true,
                preferPerformanceMode = config.image.preferPerformanceMode,
                loraModelDir = null,
                loraApplyMode = params.loraApplyMode,
            )
        return DiffusionExecutionPlan.Sequential(
            conditioningRequest =
                PlannedDiffusionRuntimeRequest(
                    spec =
                        DiffusionRuntimeSpec(
                            role = DiffusionRuntimeRole.IMAGE,
                            model = encoderSpec,
                            encoderOnly = true,
                        ),
                    options =
                        baseOptions.copy(
                            nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
                            // The conditioning pass is one 512-token forward; on a GPU backend the
                            // multi-GB encoder gets a second copy in (UMA) device memory that
                            // keepClipOnCpu does not cover — measured on S22: kernel OOM-kill at
                            // ~2.3 GB RSS entering the Qwen forward. CPU keeps sequential mode's
                            // peak-RAM promise, which is this plan's entire purpose.
                            allowGpu = false,
                        ),
                ),
            diffusionRequest =
                PlannedDiffusionRuntimeRequest(
                    spec =
                        DiffusionRuntimeSpec(
                            role = DiffusionRuntimeRole.IMAGE,
                            model = ditModel,
                            vae = params.vae,
                            textEncoder = null,
                            splitDiffusionModel = true,
                        ),
                    options = baseOptions,
                ),
        )
    }

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

    fun videoPlan(
        params: VideoGenerationRequest,
        config: LLMEdgeConfig,
    ): DiffusionExecutionPlan =
        if (params.forceSequentialLoad) {
            DiffusionExecutionPlan.Sequential(
                conditioningRequest = sequentialVideoConditioningRequest(params, config),
                diffusionRequest = sequentialVideoDiffusionRequest(params, config),
            )
        } else {
            DiffusionExecutionPlan.Direct(
                request = directVideoRequest(params, config),
            )
        }

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
        val allowGpu = params.taehv == null && config.image.useVulkan
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
