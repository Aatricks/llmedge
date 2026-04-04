package io.aatricks.llmedge.text

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.ManagedRuntimeBase
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.core.runtime.createCachedRuntimePool
import io.aatricks.llmedge.core.runtime.runtimePoolProfile
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.text.runtime.SmolLM

internal class ManagedTextModel(
    val fileSizeBytes: Long,
    val model: SmolLM,
) : ManagedRuntimeBase() {
    private fun estimatedNativeMemoryBytes(): Long =
        maxOf(
            model.getEstimatedNativeMemoryBytes().takeIf { it > 0L } ?: 0L,
            fileSizeBytes + model.getEstimatedStateMemoryBytes().coerceAtLeast(0L),
            fileSizeBytes,
        )

    override fun estimatedSizeBytes(): Long = estimatedNativeMemoryBytes()

    fun ensureOpen() {
        ensureOpen("Text runtime")
    }

    override fun close() {
        closeOnce(model::close)
    }
}

internal fun createTextRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    resolver: ModelRepository,
): RuntimePool<ModelSpec, TextModelOptions, ManagedTextModel> =
    createCachedRuntimePool(
        context = context,
        scope = scope,
        profile =
            runtimePoolProfile(
                cacheConfig = config.text.cache,
                cacheKeyPrefix = { spec, options ->
                    RuntimeCacheKeyBuilder.prefix(
                        spec.cacheKey,
                        "ctx=${options.contextSize ?: config.text.contextSize ?: 0L}",
                        "threads=${options.numThreads ?: config.text.promptThreads}",
                        "genThreads=${options.generationThreads ?: options.numThreads ?: config.text.generationThreads}",
                        "mmap=${options.useMmap ?: config.text.useMmap}",
                        "mlock=${options.useMlock ?: config.text.useMlock}",
                        "flash=${options.useFlashAttention ?: config.text.useFlashAttention}",
                    )
                },
                loadRuntime = { spec, options, backend ->
                    val modelFile = resolver.resolve(context, spec)
                    val smol = SmolLM(useVulkan = backend == ComputeBackend.VULKAN)
                    smol.load(
                        modelFile.absolutePath,
                        options.toInferenceParams(config),
                        preferredBackend = backend,
                    )
                    ManagedTextModel(fileSizeBytes = modelFile.length(), model = smol)
                },
                activeBackend = { it.model.getActiveBackend() },
                candidateRequest = { options ->
                    BackendCandidateResolver.Request(
                        subsystem = ComputeSubsystem.TEXT,
                        allowGpu = options.useVulkan ?: config.text.useVulkan,
                        openClAvailable = SmolLM.isOpenClAvailable(),
                        vulkanAvailable = SmolLM.isVulkanBackendAvailable(),
                    )
                },
            ),
    )
