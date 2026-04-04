package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.RuntimeCacheConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.ManagedRuntimeBase
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.createCachedRuntimePool
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.text.runtime.SmolLM

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
) : ManagedRuntimeBase() {
    override fun estimatedSizeBytes(): Long =
        maxOf(
            fileSizeBytes,
            fileSizeBytes + smol.getEstimatedStateMemoryBytes().coerceAtLeast(0L),
            smol.getEstimatedNativeMemoryBytes().coerceAtLeast(0L),
        )

    override fun close() {
        closeOnce {
            try {
                projector.close()
            } finally {
                smol.close()
            }
        }
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
        cacheConfig = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = config.text.cache.maxMemoryMb),
        cacheKeyPrefix = { spec, options ->
            RuntimeCacheKeyBuilder.prefix(
                spec.model.cacheKey,
                spec.projector.cacheKey,
                "threads=${options.numThreads}",
                "genThreads=${options.generationThreads}",
            )
        },
        loadRuntime = { spec, options, backend ->
            val modelFile = resolver.resolve(context, spec.model)
            val projectorFile = resolver.resolve(context, spec.projector)
            check(
                VisionPromptSupport.isReadyForMultimodalInference(spec.model, spec.projector) ||
                    VisionPromptSupport.isReadyForMultimodalInference(
                        modelFile.absolutePath,
                        projectorFile.absolutePath,
                    ),
            ) {
                VisionPromptSupport.unsupportedReason(spec.model, spec.projector)
            }
            val smol = smolLmFactory(backend == ComputeBackend.VULKAN)
            smol.load(
                modelPath = modelFile.absolutePath,
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
                preferredBackend = backend,
            )

            val projector = projectorFactory()
            projector.init(projectorFile.absolutePath, smol.getNativeModelPointer())
            check(projector.isReady()) {
                "Native projector initialization failed for ${projectorFile.name}. Ensure the mmproj file matches the selected model and that projector bindings are available."
            }

            ManagedVisionRuntime(
                fileSizeBytes = modelFile.length() + projectorFile.length(),
                smol = smol,
                projector = projector,
            )
        },
        activeBackend = { it.smol.getActiveBackend() },
        candidateRequest = {
            BackendCandidateResolver.Request(
                subsystem = io.aatricks.llmedge.runtime.ComputeSubsystem.TEXT,
                allowGpu = config.text.useVulkan,
                openClAvailable = SmolLM.isOpenClAvailable(),
                vulkanAvailable = SmolLM.isVulkanBackendAvailable(),
            )
        },
    )
