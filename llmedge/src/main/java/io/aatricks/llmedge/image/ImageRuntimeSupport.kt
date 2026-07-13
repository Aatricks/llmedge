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
    val taehv: ModelSpec? = null,
    val diffusionModelOnly: Boolean = false,
    // FLUX.2 Klein split model: route [model] to diffusion_model_path and [textEncoder] (Qwen3)
    // to llm_path instead of the default model_path / t5xxl_path slots.
    val splitDiffusionModel: Boolean = false,
    // FLUX.2 sequential mode: load ONLY [model] (the Qwen3 encoder) via llm_path, no DiT/VAE, so
    // the precompute phase peaks at the encoder size.
    val encoderOnly: Boolean = false,
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
        val resolvedTaehv = spec.taehv?.let { resolver.resolve(context, it) }
        return loadManagedModel(
            spec = spec,
            options = options,
            backend = backend,
            resolvedModel = resolvedModel,
            resolvedVae = resolvedVae,
            resolvedTextEncoder = resolvedTextEncoder,
            resolvedTaehv = resolvedTaehv,
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
        resolvedTaehv: File?,
    ): ManagedDiffusionModel {
        val fileSizeBytes =
            estimateFileSizeBytes(
                resolvedModel,
                resolvedVae,
                resolvedTextEncoder,
                resolvedTaehv,
            )
        val preferredFlash = options.flashAttn
        try {
            return createManagedModel(
                options = options,
                backend = backend,
                resolvedModel = resolvedModel,
                resolvedVae = resolvedVae,
                resolvedTextEncoder = resolvedTextEncoder,
                resolvedTaehv = resolvedTaehv,
                fileSizeBytes = fileSizeBytes,
                flashAttn = preferredFlash,
                diffusionModelOnly = spec.diffusionModelOnly,
                splitDiffusionModel = spec.splitDiffusionModel,
                encoderOnly = spec.encoderOnly,
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
                    resolvedTaehv = resolvedTaehv,
                    fileSizeBytes = fileSizeBytes,
                    flashAttn = false,
                    diffusionModelOnly = spec.diffusionModelOnly,
                    splitDiffusionModel = spec.splitDiffusionModel,
                    encoderOnly = spec.encoderOnly,
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
        resolvedTaehv: File?,
        fileSizeBytes: Long,
        flashAttn: Boolean,
        diffusionModelOnly: Boolean,
        splitDiffusionModel: Boolean,
        encoderOnly: Boolean,
    ): ManagedDiffusionModel {
        AndroidLogAdapter.i(
            LOG_TAG,
            "Creating managed ${backend.name} runtime for ${resolvedModel.name} flash=$flashAttn sequential=${options.sequentialLoad}",
        )
        phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.LOADING, backend.name)
        val model =
            StableDiffusion.loadWithRuntimeBackend(
                context = context,
                // encoderOnly: load just the Qwen3 encoder via llm_path (no model/diffusion/vae).
                modelPath = if (splitDiffusionModel || diffusionModelOnly || encoderOnly) null else resolvedModel.absolutePath,
                vaePath = if (encoderOnly) null else resolvedVae?.absolutePath,
                t5xxlPath = if (splitDiffusionModel || encoderOnly) null else resolvedTextEncoder?.absolutePath,
                taesdPath = if (encoderOnly) null else resolvedTaehv?.absolutePath,
                diffusionModelPath = if (splitDiffusionModel || diffusionModelOnly) resolvedModel.absolutePath else null,
                llmPath =
                    when {
                        encoderOnly -> resolvedModel.absolutePath
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
                        spec.taehv?.cacheKey,
                        "diffusionOnly=${spec.diffusionModelOnly}",
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
