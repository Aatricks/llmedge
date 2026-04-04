package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.RuntimeCacheConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.BackendPolicy
import io.aatricks.llmedge.core.runtime.CachedRuntimeDescriptor
import io.aatricks.llmedge.core.runtime.ManagedRuntime
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.core.runtime.RuntimeKeyStrategy
import io.aatricks.llmedge.core.runtime.RuntimeLoader
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.createCachedRuntimePool
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.sync.Mutex

internal data class VisionRuntimeSpec(
    val model: ModelSpec,
    val projector: ModelSpec,
)

internal data class VisionLoadOptions(
    val numThreads: Int,
    val generationThreads: Int,
)

internal class ManagedVisionRuntime(
    private val fileSizeBytes: Long,
    val smol: SmolLM,
    val projector: Projector,
) : ManagedRuntime {
    override val mutex: Mutex = Mutex()

    override fun estimatedSizeBytes(): Long =
        maxOf(
            fileSizeBytes,
            fileSizeBytes + smol.getEstimatedStateMemoryBytes().coerceAtLeast(0L),
            smol.getEstimatedNativeMemoryBytes().coerceAtLeast(0L),
        )

    override fun close() {
        try {
            projector.close()
        } finally {
            smol.close()
        }
    }
}

internal class VisionRuntimeKeyStrategy : RuntimeKeyStrategy<VisionRuntimeSpec, VisionLoadOptions> {
    override fun prefix(
        spec: VisionRuntimeSpec,
        options: VisionLoadOptions,
    ): String =
        RuntimeCacheKeyBuilder.prefix(
            spec.model.cacheKey,
            spec.projector.cacheKey,
            "threads=${options.numThreads}",
            "genThreads=${options.generationThreads}",
        )
}

internal class VisionBackendPolicy(
    private val config: LLMEdgeConfig,
) : BackendPolicy<VisionLoadOptions> {
    override fun request(options: VisionLoadOptions) =
        BackendCandidateResolver.Request(
            subsystem = ComputeSubsystem.TEXT,
            allowGpu = config.text.useVulkan,
            openClAvailable = SmolLM.isOpenClAvailable(),
            vulkanAvailable = SmolLM.isVulkanBackendAvailable(),
        )
}

internal class VisionRuntimeLoader(
    private val context: Context,
    private val resolver: ModelRepository,
    private val config: LLMEdgeConfig,
    private val smolLmFactory: (Boolean) -> SmolLM,
    private val projectorFactory: () -> Projector,
) : RuntimeLoader<VisionRuntimeSpec, VisionLoadOptions, ManagedVisionRuntime> {
    override suspend fun load(
        spec: VisionRuntimeSpec,
        options: VisionLoadOptions,
    ): ManagedVisionRuntime {
        val modelFile = resolver.resolve(context, spec.model)
        val projectorFile = resolver.resolve(context, spec.projector)
        val smol = smolLmFactory(config.text.useVulkan)
        val adapter = SmolLMVisionAdapter(context, smol)
        adapter.loadVisionModel(
            modelPath = modelFile.absolutePath,
            mmprojPath = projectorFile.absolutePath,
            params =
                SmolLM.InferenceParams(
                    numThreads = options.numThreads.coerceAtLeast(1),
                    generationThreads = options.generationThreads.coerceAtLeast(1),
                    contextSize = null,
                    storeChats = false,
                    temperature = 0.0f,
                    useFlashAttn = config.text.useFlashAttention,
                    thinkingMode = SmolLM.ThinkingMode.DEFAULT,
                ),
        )

        val projector = projectorFactory()
        projector.init(projectorFile.absolutePath, smol.getNativeModelPointer())
        check(projector.isReady()) {
            "Native projector initialization failed for ${projectorFile.name}. Ensure the mmproj file matches the selected model and that projector bindings are available."
        }

        return ManagedVisionRuntime(
            fileSizeBytes = modelFile.length() + projectorFile.length(),
            smol = smol,
            projector = projector,
        )
    }
}

internal fun createVisionRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    resolver: ModelRepository,
    config: LLMEdgeConfig,
    smolLmFactory: (Boolean) -> SmolLM,
    projectorFactory: () -> Projector,
): RuntimePool<VisionRuntimeSpec, VisionLoadOptions, ManagedVisionRuntime> =
    createCachedRuntimePool(
        context = context,
        scope = scope,
        descriptor =
            CachedRuntimeDescriptor(
                cache = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = config.text.cache.maxMemoryMb),
                keyStrategy = VisionRuntimeKeyStrategy(),
                runtimeLoader = VisionRuntimeLoader(context, resolver, config, smolLmFactory, projectorFactory),
                activeBackend = { ComputeBackend.CPU },
                backendPolicy = VisionBackendPolicy(config),
            ),
    )
