package io.aatricks.llmedge.image

import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.model.ModelSpec
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
        val conditioningRequest2: PlannedDiffusionRuntimeRequest? = null,
        val diffusionRequest: PlannedDiffusionRuntimeRequest,
    ) : DiffusionExecutionPlan
}

internal object ImageRuntimeRequestPlanner {
    private fun isMiniT2ILarge(model: ModelSpec?): Boolean {
        return model is ModelSpec.HuggingFace &&
            model.repoId == "MiniT2I/MiniT2I" &&
            model.filename == "minit2i-l-16/transformer/diffusion_pytorch_model.safetensors"
    }

    fun imageRequest(
        params: ImageGenerationRequest,
        config: LLMEdgeConfig,
    ): PlannedDiffusionRuntimeRequest {
        val modelSpec = params.model ?: config.models.image
        val isLarge = isMiniT2ILarge(modelSpec)
        return PlannedDiffusionRuntimeRequest(
            spec =
                DiffusionRuntimeSpec(
                    role = DiffusionRuntimeRole.IMAGE,
                    model = modelSpec,
                    vae = params.vae,
                    textEncoder = params.textEncoder,
                    t5xxl = params.t5xxl,
                    clipL = params.clipL,
                    clipG = params.clipG,
                    clipVision = params.clipVision,
                    llmVision = params.llmVision,
                    controlNet = params.controlNet,
                    photoMaker = params.photoMaker,
                    embeddingsConnectors = params.embeddingsConnectors,
                    diffusionModelOnly = params.diffusionModelOnly,
                    splitDiffusionModel = params.splitDiffusionModel,
                    conditioningProfile = ImageExecutionPlanner.recipeFor(params).profile,
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
                    weightType = if (isLarge) "q8_0" else null,
                    tensorTypeRules = if (isLarge) ".*mask_token.*=f16" else null,
                ),
        )
    }

    /**
     * Two-phase plan for FLUX.2 sequential low-memory generation: phase 1 loads ONLY the Qwen3
     * encoder (to precompute the conditioning), phase 2 loads ONLY the DiT (+VAE). Peak RAM is the
     * larger of the two phases, not their sum.
     */
    fun imageSequentialPlan(
        params: ImageGenerationRequest,
        config: LLMEdgeConfig,
        profile: ImageConditioningProfile = ImageExecutionPlanner.recipeFor(params).profile,
    ): DiffusionExecutionPlan.Sequential {
        require(profile != ImageConditioningProfile.NONE) {
            "Sequential image generation requires a splittable conditioning profile"
        }
        val isSd3 = profile == ImageConditioningProfile.SD3_CLIP_T5
        val isMaskedT5 = profile == ImageConditioningProfile.MASKED_T5
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
        val conditioningSpec: DiffusionRuntimeSpec
        val conditioningSpec2: DiffusionRuntimeSpec?
        if (isSd3) {
            val clipL = params.clipL ?: error("SD3 sequential conditioning requires a clipL model")
            val t5xxl = params.t5xxl ?: error("SD3 sequential conditioning requires a t5xxl model")
            conditioningSpec = DiffusionRuntimeSpec(
                role = DiffusionRuntimeRole.IMAGE,
                model = clipL,
                t5xxl = null,
                clipL = clipL,
                clipG = params.clipG,
                encoderOnly = true,
            )
            conditioningSpec2 = DiffusionRuntimeSpec(
                role = DiffusionRuntimeRole.IMAGE,
                model = t5xxl,
                t5xxl = t5xxl,
                clipL = null,
                clipG = null,
                encoderOnly = true,
            )
        } else {
            val encoderSpec = params.textEncoder ?: error("sequential image generation requires a textEncoder")
            conditioningSpec = DiffusionRuntimeSpec(
                role = DiffusionRuntimeRole.IMAGE,
                model = encoderSpec,
                encoderOnly = true,
                conditioningProfile = profile,
            )
            conditioningSpec2 = null
        }
        val diffusionSpec =
            when {
                isSd3 ->
                    DiffusionRuntimeSpec(
                        role = DiffusionRuntimeRole.IMAGE,
                        model = ditModel,
                        vae = params.vae,
                        splitDiffusionModel = true,
                        conditioningProfile = profile,
                    )
                isMaskedT5 ->
                    DiffusionRuntimeSpec(
                        role = DiffusionRuntimeRole.IMAGE,
                        model = ditModel,
                        vae = params.vae,
                        diffusionModelOnly = true,
                        conditioningProfile = profile,
                    )
                else ->
                    DiffusionRuntimeSpec(
                        role = DiffusionRuntimeRole.IMAGE,
                        model = ditModel,
                        vae = params.vae,
                        textEncoder = null,
                        splitDiffusionModel = true,
                        conditioningProfile = profile,
                    )
            }
        val conditioningOptions = baseOptions.copy(
            nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
            allowGpu = false,
        )
        val t5ConditioningOptions = conditioningOptions.copy(
            allowGpu = config.image.useVulkan,
        )
        // The conditioning runtime is gone before diffusion begins, so keep diffusion weights
        // on a supported GPU instead of mirroring its multi-GB parameters in system RAM.
        val isMiniT2ILarge = isMiniT2ILarge(ditModel)
        val isMaskedT5Profile = profile == ImageConditioningProfile.MASKED_T5
        val diffusionOptions = baseOptions.copy(
            offloadToCpu = false,
            weightType = if (isMiniT2ILarge && isMaskedT5Profile) "q8_0" else null,
            tensorTypeRules = if (isMiniT2ILarge && isMaskedT5Profile) ".*mask_token.*=f16" else null,
        )
        return DiffusionExecutionPlan.Sequential(
            conditioningRequest =
                PlannedDiffusionRuntimeRequest(
                    spec = conditioningSpec,
                    options = conditioningOptions,
                ),
            conditioningRequest2 = conditioningSpec2?.let {
                PlannedDiffusionRuntimeRequest(
                    spec = it,
                    options = t5ConditioningOptions,
                )
            },
            diffusionRequest =
                PlannedDiffusionRuntimeRequest(
                    spec = diffusionSpec,
                    options = diffusionOptions,
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
            highNoiseDiffusionModel = params.highNoiseDiffusionModel,
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
