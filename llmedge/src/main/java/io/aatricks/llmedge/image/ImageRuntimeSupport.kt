package io.aatricks.llmedge.image

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.ManagedRuntimeBase
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.core.runtime.RuntimeCapabilities
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.createCachedRuntimePool
import io.aatricks.llmedge.core.runtime.runtimePoolProfile
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths
import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import java.io.File

internal enum class DiffusionRuntimeRole {
    IMAGE,
    VIDEO,
    VIDEO_TEXT_ENCODER,
}

internal data class DiffusionRuntimeSpec(
    val role: DiffusionRuntimeRole,
    val model: ModelSpec,
    val vae: ModelSpec? = null,
    val textEncoder: ModelSpec? = null,
    val t5xxl: ModelSpec? = null,
    val taehv: ModelSpec? = null,
    val clipL: ModelSpec? = null,
    val clipG: ModelSpec? = null,
    val clipVision: ModelSpec? = null,
    val llmVision: ModelSpec? = null,
    val controlNet: ModelSpec? = null,
    val photoMaker: ModelSpec? = null,
    val embeddingsConnectors: ModelSpec? = null,
    val highNoiseDiffusionModel: ModelSpec? = null,
    val diffusionModelOnly: Boolean = false,
    // FLUX.2 Klein split model: route [model] to diffusion_model_path and [textEncoder] (Qwen3)
    // to llm_path instead of the default model_path / t5xxl_path slots.
    val splitDiffusionModel: Boolean = false,
    // FLUX.2 / SD3 sequential mode: load ONLY the encoder(s) via llm_path / componentPaths, no DiT/VAE,
    // so the precompute phase peaks at the encoder size.
    val encoderOnly: Boolean = false,
    val conditioningProfile: ImageConditioningProfile = ImageConditioningProfile.NONE,
)

internal data class DiffusionLoadOptions(
    val subsystem: ComputeSubsystem,
    val allowGpu: Boolean,
    val nThreads: Int,
    val offloadToCpu: Boolean,
    val keepClipOnCpu: Boolean,
    val keepVaeOnCpu: Boolean,
    val flashAttn: Boolean,
    val vaeDecodeOnly: Boolean = true,
    val sequentialLoad: Boolean? = null,
    val preferPerformanceMode: Boolean,
    val flowShift: Float = Float.POSITIVE_INFINITY,
    val loraModelDir: String? = null,
    val loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
)

internal class ManagedDiffusionModel(
    val fileSizeBytes: Long,
    val backend: ComputeBackend,
    val flashAttnEnabled: Boolean,
    val model: StableDiffusion,
) : ManagedRuntimeBase() {
    override fun estimatedSizeBytes(): Long = fileSizeBytes

    override fun close() {
        closeOnce(model::close)
    }
}

internal class DiffusionRuntimeLoader(
    private val context: Context,
    private val resolver: ModelRepository,
    private val phaseListener: DiffusionPhaseListener? = null,
) {
    companion object {
        private const val LOG_TAG = "DiffusionRuntimeLoader"
    }

    suspend fun load(
        spec: DiffusionRuntimeSpec,
        options: DiffusionLoadOptions,
        backend: ComputeBackend,
    ): ManagedDiffusionModel {
        // RESOLVING may legitimately sit CPU-flat for minutes (network download); the watchdog
        // exempts it from the flat-CPU hang rule, so it must be distinguishable from LOADING.
        phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.RESOLVING_MODEL, backend.name)
        val resolvedModel = resolver.resolve(context, spec.model)
        val resolvedVae = spec.vae?.let { resolver.resolve(context, it) }
        val resolvedTextEncoder = spec.textEncoder?.let { resolver.resolve(context, it) }
        val resolvedT5xxl = spec.t5xxl?.let { resolver.resolve(context, it) }
        val resolvedTaehv = spec.taehv?.let { resolver.resolve(context, it) }
        val resolvedClipL = spec.clipL?.let { resolver.resolve(context, it) }
        val resolvedClipG = spec.clipG?.let { resolver.resolve(context, it) }
        val resolvedClipVision = spec.clipVision?.let { resolver.resolve(context, it) }
        val resolvedLlmVision = spec.llmVision?.let { resolver.resolve(context, it) }
        val resolvedControlNet = spec.controlNet?.let { resolver.resolve(context, it) }
        val resolvedPhotoMaker = spec.photoMaker?.let { resolver.resolve(context, it) }
        val resolvedEmbeddingsConnectors = spec.embeddingsConnectors?.let { resolver.resolve(context, it) }
        val resolvedHighNoiseDiffusionModel = spec.highNoiseDiffusionModel?.let { resolver.resolve(context, it) }
        return loadManagedModel(
            spec = spec,
            options = options,
            backend = backend,
            resolvedModel = resolvedModel,
            resolvedVae = resolvedVae,
            resolvedTextEncoder = resolvedTextEncoder,
            resolvedT5xxl = resolvedT5xxl,
            resolvedTaehv = resolvedTaehv,
            resolvedClipL = resolvedClipL,
            resolvedClipG = resolvedClipG,
            resolvedClipVision = resolvedClipVision,
            resolvedLlmVision = resolvedLlmVision,
            resolvedControlNet = resolvedControlNet,
            resolvedPhotoMaker = resolvedPhotoMaker,
            resolvedEmbeddingsConnectors = resolvedEmbeddingsConnectors,
            resolvedHighNoiseDiffusionModel = resolvedHighNoiseDiffusionModel,
        )
    }

    private fun estimateFileSizeBytes(vararg files: File?): Long =
        files.filterNotNull().sumOf(File::length)

    private suspend fun loadManagedModel(
        spec: DiffusionRuntimeSpec,
        options: DiffusionLoadOptions,
        backend: ComputeBackend,
        resolvedModel: File,
        resolvedVae: File?,
        resolvedTextEncoder: File?,
        resolvedT5xxl: File?,
        resolvedTaehv: File?,
        resolvedClipL: File?,
        resolvedClipG: File?,
        resolvedClipVision: File?,
        resolvedLlmVision: File?,
        resolvedControlNet: File?,
        resolvedPhotoMaker: File?,
        resolvedEmbeddingsConnectors: File?,
        resolvedHighNoiseDiffusionModel: File?,
    ): ManagedDiffusionModel {
        val fileSizeBytes =
            estimateFileSizeBytes(
                resolvedModel,
                resolvedVae,
                resolvedTextEncoder,
                resolvedT5xxl,
                resolvedTaehv,
                resolvedClipL,
                resolvedClipG,
                resolvedClipVision,
                resolvedLlmVision,
                resolvedControlNet,
                resolvedPhotoMaker,
                resolvedEmbeddingsConnectors,
                resolvedHighNoiseDiffusionModel,
            )
        val preferredFlash = options.flashAttn
        val diffusionModelOnly =
            spec.diffusionModelOnly ||
                (spec.role != DiffusionRuntimeRole.IMAGE &&
                    spec.model.hints.artifactKind == ModelArtifactKind.DIFFUSION_MODEL)
        val miniT2iConditionerOnly =
            spec.encoderOnly && spec.conditioningProfile == ImageConditioningProfile.MASKED_T5
        try {
            return createManagedModel(
                options = options,
                backend = backend,
                resolvedModel = resolvedModel,
                resolvedVae = resolvedVae,
                resolvedTextEncoder = resolvedTextEncoder,
                resolvedT5xxl = resolvedT5xxl,
                resolvedTaehv = resolvedTaehv,
                resolvedClipL = resolvedClipL,
                resolvedClipG = resolvedClipG,
                resolvedClipVision = resolvedClipVision,
                resolvedLlmVision = resolvedLlmVision,
                resolvedControlNet = resolvedControlNet,
                resolvedPhotoMaker = resolvedPhotoMaker,
                resolvedEmbeddingsConnectors = resolvedEmbeddingsConnectors,
                resolvedHighNoiseDiffusionModel = resolvedHighNoiseDiffusionModel,
                fileSizeBytes = fileSizeBytes,
                flashAttn = preferredFlash,
                diffusionModelOnly = diffusionModelOnly,
                splitDiffusionModel = spec.splitDiffusionModel,
                encoderOnly = spec.encoderOnly,
                miniT2iConditionerOnly = miniT2iConditionerOnly,
            )
        } catch (error: Throwable) {
            if (!shouldRetryWithoutFlash(spec, options)) {
                throw error
            }
            AndroidLogAdapter.w(
                LOG_TAG,
                "Failed to load ${spec.role} runtime on $backend with flash attention; retrying once with flash attention disabled",
            )
            try {
                return createManagedModel(
                    options = options,
                    backend = backend,
                    resolvedModel = resolvedModel,
                    resolvedVae = resolvedVae,
                    resolvedTextEncoder = resolvedTextEncoder,
                    resolvedT5xxl = resolvedT5xxl,
                    resolvedTaehv = resolvedTaehv,
                    resolvedClipL = resolvedClipL,
                    resolvedClipG = resolvedClipG,
                    resolvedClipVision = resolvedClipVision,
                    resolvedLlmVision = resolvedLlmVision,
                    resolvedControlNet = resolvedControlNet,
                    resolvedPhotoMaker = resolvedPhotoMaker,
                    resolvedEmbeddingsConnectors = resolvedEmbeddingsConnectors,
                    resolvedHighNoiseDiffusionModel = resolvedHighNoiseDiffusionModel,
                    fileSizeBytes = fileSizeBytes,
                    flashAttn = false,
                    diffusionModelOnly = diffusionModelOnly,
                    splitDiffusionModel = spec.splitDiffusionModel,
                    encoderOnly = spec.encoderOnly,
                    miniT2iConditionerOnly = miniT2iConditionerOnly,
                )
            } catch (fallbackError: Throwable) {
                fallbackError.addSuppressed(error)
                throw fallbackError
            }
        }
    }

    private suspend fun createManagedModel(
        options: DiffusionLoadOptions,
        backend: ComputeBackend,
        resolvedModel: File,
        resolvedVae: File?,
        resolvedTextEncoder: File?,
        resolvedT5xxl: File?,
        resolvedTaehv: File?,
        resolvedClipL: File?,
        resolvedClipG: File?,
        resolvedClipVision: File?,
        resolvedLlmVision: File?,
        resolvedControlNet: File?,
        resolvedPhotoMaker: File?,
        resolvedEmbeddingsConnectors: File?,
        resolvedHighNoiseDiffusionModel: File?,
        fileSizeBytes: Long,
        flashAttn: Boolean,
        diffusionModelOnly: Boolean,
        splitDiffusionModel: Boolean,
        encoderOnly: Boolean,
        miniT2iConditionerOnly: Boolean,
    ): ManagedDiffusionModel {
        AndroidLogAdapter.i(
            LOG_TAG,
            "Creating managed ${backend.name} runtime for ${resolvedModel.name} flash=$flashAttn sequential=${options.sequentialLoad}",
        )
        phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.LOADING, backend.name)
        val isSd3EncoderOnly = encoderOnly && (
            (resolvedClipL != null && resolvedClipG != null && resolvedT5xxl == null) ||
            (resolvedClipL == null && resolvedClipG == null && resolvedT5xxl != null)
        )
        val componentPaths = if (encoderOnly && !isSd3EncoderOnly && !miniT2iConditionerOnly) {
            null
        } else {
            val paths = StableDiffusionComponentPaths(
                clipLPath = resolvedClipL?.absolutePath,
                clipGPath = resolvedClipG?.absolutePath,
                clipVisionPath = resolvedClipVision?.absolutePath,
                llmVisionPath = resolvedLlmVision?.absolutePath,
                highNoiseDiffusionModelPath = resolvedHighNoiseDiffusionModel?.absolutePath,
                embeddingsConnectorsPath = resolvedEmbeddingsConnectors?.absolutePath,
                audioVaePath = null,
                controlNetPath = resolvedControlNet?.absolutePath,
                photoMakerPath = resolvedPhotoMaker?.absolutePath,
                miniT2iConditionerOnly = miniT2iConditionerOnly,
            )
            if (paths.isAllNull()) null else paths
        }
        val model =
            StableDiffusion.loadWithRuntimeBackend(
                context = context,
                // encoderOnly: load just the text encoder(s) (Qwen3 via llmPath for FLUX; CLIP-L/CLIP-G/T5XXL for SD3).
                modelPath =
                    if (splitDiffusionModel || diffusionModelOnly || (encoderOnly && !miniT2iConditionerOnly)) {
                        null
                    } else {
                        resolvedModel.absolutePath
                    },
                vaePath = if (encoderOnly) null else resolvedVae?.absolutePath,
                t5xxlPath = resolvedT5xxl?.absolutePath ?: (if (splitDiffusionModel || (encoderOnly && !isSd3EncoderOnly)) null else resolvedTextEncoder?.absolutePath),
                taesdPath = if (encoderOnly) null else resolvedTaehv?.absolutePath,
                diffusionModelPath = if (splitDiffusionModel || diffusionModelOnly) resolvedModel.absolutePath else null,
                llmPath =
                    when {
                        encoderOnly && !isSd3EncoderOnly && !miniT2iConditionerOnly -> resolvedModel.absolutePath
                        splitDiffusionModel -> resolvedTextEncoder?.absolutePath
                        else -> null
                    },
                nThreads = options.nThreads,
                offloadToCpu = options.offloadToCpu,
                keepClipOnCpu = options.keepClipOnCpu,
                keepVaeOnCpu = options.keepVaeOnCpu,
                flashAttn = flashAttn,
                vaeDecodeOnly = options.vaeDecodeOnly,
                sequentialLoad = options.sequentialLoad,
                preferPerformanceMode = options.preferPerformanceMode,
                flowShift = options.flowShift,
                loraModelDir = options.loraModelDir,
                loraApplyMode = options.loraApplyMode,
                preferredBackend = backend,
                componentPaths = componentPaths,
            )
        AndroidLogAdapter.i(
            LOG_TAG,
            "Managed ${backend.name} runtime ready for ${resolvedModel.name}",
        )
        return ManagedDiffusionModel(
            fileSizeBytes = fileSizeBytes,
            backend = backend,
            flashAttnEnabled = flashAttn,
            model = model,
        )
    }

    private fun shouldRetryWithoutFlash(
        spec: DiffusionRuntimeSpec,
        options: DiffusionLoadOptions,
    ): Boolean =
        spec.role == DiffusionRuntimeRole.IMAGE &&
            options.flashAttn &&
            options.sequentialLoad != true
}

internal fun createDiffusionRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    resolver: ModelRepository,
    phaseListener: DiffusionPhaseListener? = null,
): RuntimePool<DiffusionRuntimeSpec, DiffusionLoadOptions, ManagedDiffusionModel> =
    createCachedRuntimePool(
        context = context,
        scope = scope,
        profile =
            runtimePoolProfile(
                cacheConfig = config.image.cache,
                cacheKeyPrefix = { spec, options ->
                    RuntimeCacheKeyBuilder.prefix(
                        "role=${spec.role.name}",
                        spec.model.cacheKey,
                        spec.vae?.cacheKey,
                        spec.textEncoder?.cacheKey,
                        spec.t5xxl?.cacheKey,
                        spec.taehv?.cacheKey,
                        spec.clipL?.cacheKey,
                        spec.clipG?.cacheKey,
                        spec.clipVision?.cacheKey,
                        spec.llmVision?.cacheKey,
                        spec.controlNet?.cacheKey,
                        spec.photoMaker?.cacheKey,
                        spec.embeddingsConnectors?.cacheKey,
                        spec.highNoiseDiffusionModel?.cacheKey,
                        "diffusionOnly=${spec.diffusionModelOnly}",
                        "conditioning=${spec.conditioningProfile.name}",
                        "threads=${options.nThreads}",
                        "gpu=${options.allowGpu}",
                        "offload=${options.offloadToCpu}",
                        "clipCpu=${options.keepClipOnCpu}",
                        "vaeCpu=${options.keepVaeOnCpu}",
                        "flash=${options.flashAttn}",
                        "vaeDecodeOnly=${options.vaeDecodeOnly}",
                        "sequential=${options.sequentialLoad}",
                        "perf=${options.preferPerformanceMode}",
                        "flowShift=${options.flowShift}",
                        "loraDir=${options.loraModelDir}",
                        "loraMode=${options.loraApplyMode.id}",
                    )
                },
                loadRuntime = DiffusionRuntimeLoader(context, resolver, phaseListener)::load,
                activeBackend = { it.backend },
                candidateRequest = { options ->
                    BackendCandidateResolver.Request(
                        subsystem = options.subsystem,
                        allowGpu = options.allowGpu,
                        openClAvailable = RuntimeCapabilities.isStableDiffusionOpenClAvailable(),
                        vulkanAvailable = RuntimeCapabilities.isStableDiffusionVulkanAvailable(),
                    )
                },
            ),
    )
