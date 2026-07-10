package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.ManagedRuntimeBase
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.createCachedRuntimePool
import io.aatricks.llmedge.core.runtime.runtimePoolProfile
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.text.runtime.SmolLM

/**
 * Micro-batch size for vision runtimes. Non-causal projectors (Gemma3-family)
 * require the entire image chunk in one micro-batch, and batched decode of the
 * embedding fallback path uses the same bound.
 */
internal const val VISION_UBATCH = 1024

internal data class VisionRuntimeSpec(
    val model: ModelSpec,
    val projector: ModelSpec,
)

internal data class VisionLoadOptions(
    val promptThreads: Int,
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
        profile =
            runtimePoolProfile(
                cacheConfig = config.vision.cache,
                cacheKeyPrefix = { spec, options ->
                    RuntimeCacheKeyBuilder.prefix(
                        spec.model.cacheKey,
                        spec.projector.cacheKey,
                        "threads=${options.promptThreads}",
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
                    var loadedSmol: SmolLM? = null
                    var initializedProjector: Projector? = null
                    try {
                        smol.load(
                            modelPath = modelFile.absolutePath,
                            params =
                                SmolLM.InferenceParams(
                                    numThreads = options.promptThreads.coerceAtLeast(1),
                                    generationThreads = options.generationThreads.coerceAtLeast(1),
                                    contextSize = null,
                                    storeChats = false,
                                    temperature = 0.0f,
                                    useFlashAttn = config.vision.useFlashAttention,
                                    thinkingMode = SmolLM.ThinkingMode.DEFAULT,
                                    // Non-causal projectors (Gemma3-family) require the whole
                                    // image chunk in a single micro-batch: llama_decode asserts
                                    // n_ubatch >= n_tokens when causal_attn is off. 1024 covers
                                    // every current projector's per-image token count.
                                    nUbatch = VISION_UBATCH,
                                ),
                            preferredBackend = backend,
                        )
                        loadedSmol = smol

                        val projector = projectorFactory()
                        initializedProjector = projector
                        projector.init(projectorFile.absolutePath, smol.getNativeModelPointer())
                        check(projector.isReady()) {
                            "Native projector initialization failed for ${projectorFile.name}. Ensure the mmproj file matches the selected model and that projector bindings are available."
                        }

                        val runtime = ManagedVisionRuntime(
                            fileSizeBytes = modelFile.length() + projectorFile.length(),
                            smol = smol,
                            projector = projector,
                        )
                        loadedSmol = null
                        initializedProjector = null
                        runtime
                    } catch (t: Throwable) {
                        try {
                            initializedProjector?.close()
                        } catch (e: Exception) {
                        }
                        try {
                            loadedSmol?.close()
                        } catch (e: Exception) {
                        }
                        throw t
                    }
                },
                activeBackend = { it.smol.getActiveBackend() },
                candidateRequest = {
                    BackendCandidateResolver.Request(
                        subsystem = io.aatricks.llmedge.runtime.ComputeSubsystem.VISION,
                        allowGpu = config.vision.useVulkan,
                        openClAvailable = SmolLM.isOpenClAvailable(),
                        vulkanAvailable = SmolLM.isVulkanBackendAvailable(),
                    )
                },
            ),
    )
