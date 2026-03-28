package io.aatricks.llmedge.image

import android.content.Context
import io.aatricks.llmedge.LLMEdge
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ModelCacheFactory
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.BackendPolicy
import io.aatricks.llmedge.core.runtime.ManagedRuntime
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.core.runtime.RuntimeKeyStrategy
import io.aatricks.llmedge.core.runtime.RuntimeLoader
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

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
) : ManagedRuntime {
    override val mutex: Mutex = Mutex()
    private val closed = AtomicBoolean(false)

    override fun estimatedSizeBytes(): Long = fileSizeBytes

    override fun close() {
        if (!closed.compareAndSet(false, true)) {
            return
        }
        runBlocking {
            mutex.withLock {
                model.close()
            }
        }
    }
}

internal class DiffusionRuntimeKeyStrategy : RuntimeKeyStrategy<DiffusionRuntimeSpec, DiffusionLoadOptions> {
    override fun prefix(
        spec: DiffusionRuntimeSpec,
        options: DiffusionLoadOptions,
    ): String =
        RuntimeCacheKeyBuilder.prefix(
            "role=${spec.role.name}",
            spec.model.cacheKey,
            spec.vae?.cacheKey,
            spec.textEncoder?.cacheKey,
            spec.taehv?.cacheKey,
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
}

internal class DiffusionBackendPolicy : BackendPolicy<DiffusionLoadOptions> {
    override fun request(options: DiffusionLoadOptions) =
        BackendCandidateResolver.Request(
            subsystem = options.subsystem,
            allowGpu = options.allowGpu,
            openClAvailable = StableDiffusion.isOpenClAvailable(),
            vulkanAvailable = LLMEdge.isVulkanAvailable(),
        )
}

internal class DiffusionRuntimeLoader(
    private val context: Context,
    private val resolver: ModelResolver,
) : RuntimeLoader<DiffusionRuntimeSpec, DiffusionLoadOptions, ManagedDiffusionModel> {
    companion object {
        private const val LOG_TAG = "DiffusionRuntimeLoader"
    }

    override suspend fun load(
        spec: DiffusionRuntimeSpec,
        options: DiffusionLoadOptions,
    ): ManagedDiffusionModel {
        val resolvedModel = resolver.resolve(context, spec.model)
        val resolvedVae = spec.vae?.let { resolver.resolve(context, it) }
        val resolvedTextEncoder = spec.textEncoder?.let { resolver.resolve(context, it) }
        val resolvedTaehv = spec.taehv?.let { resolver.resolve(context, it) }

        val request =
            BackendCandidateResolver.Request(
                subsystem = options.subsystem,
                allowGpu = options.allowGpu,
                openClAvailable = StableDiffusion.isOpenClAvailable(),
                vulkanAvailable = LLMEdge.isVulkanAvailable(),
            )

        var lastError: Throwable? = null
        for (backend in BackendCandidateResolver.candidates(request)) {
            try {
                return loadManagedModel(
                    spec = spec,
                    options = options,
                    backend = backend,
                    resolvedModel = resolvedModel,
                    resolvedVae = resolvedVae,
                    resolvedTextEncoder = resolvedTextEncoder,
                    resolvedTaehv = resolvedTaehv,
                )
            } catch (error: Throwable) {
                lastError = error
                if (backend == ComputeBackend.CPU) {
                    break
                }
                BackendCandidateResolver.blacklist(options.subsystem, backend)
                AndroidLogAdapter.w(
                    LOG_TAG,
                    "Failed to load ${spec.role} runtime on $backend; retrying with the next backend",
                )
            }
        }

        throw (lastError ?: IllegalStateException("Diffusion runtime load failed without a reported cause"))
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
    ): ManagedDiffusionModel {
        AndroidLogAdapter.i(
            LOG_TAG,
            "Creating managed ${backend.name} runtime for ${resolvedModel.name} flash=$flashAttn sequential=${options.sequentialLoad}",
        )
        val model =
            StableDiffusion.loadWithRuntimeBackend(
                context = context,
                modelPath = resolvedModel.absolutePath,
                vaePath = resolvedVae?.absolutePath,
                t5xxlPath = resolvedTextEncoder?.absolutePath,
                taesdPath = resolvedTaehv?.absolutePath,
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
    resolver: ModelResolver,
): RuntimePool<DiffusionRuntimeSpec, DiffusionLoadOptions, ManagedDiffusionModel> =
    RuntimePool(
        cache =
            ModelCacheFactory.create(
                context = context,
                scope = scope,
                maxCacheSize = config.image.cache.maxEntries,
                maxMemoryMB = config.image.cache.maxMemoryMb,
            ),
        keyStrategy = DiffusionRuntimeKeyStrategy(),
        runtimeLoader = DiffusionRuntimeLoader(context, resolver),
        activeBackend = { it.backend },
        backendPolicy = DiffusionBackendPolicy(),
    )
