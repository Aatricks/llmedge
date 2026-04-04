package io.aatricks.llmedge.text

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendPolicy
import io.aatricks.llmedge.core.runtime.CachedRuntimeDescriptor
import io.aatricks.llmedge.core.runtime.ManagedRuntimeBase
import io.aatricks.llmedge.core.runtime.RuntimeKeyStrategy
import io.aatricks.llmedge.core.runtime.RuntimeLoader
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.core.runtime.createCachedRuntimePool
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

internal class TextRuntimeKeyStrategy(
    private val config: LLMEdgeConfig,
) : RuntimeKeyStrategy<ModelSpec, TextModelOptions> {
    override fun prefix(
        spec: ModelSpec,
        options: TextModelOptions,
    ): String =
        RuntimeCacheKeyBuilder.prefix(
            spec.cacheKey,
            "ctx=${options.contextSize ?: config.text.contextSize ?: 0L}",
            "threads=${options.numThreads ?: config.text.promptThreads}",
            "genThreads=${options.generationThreads ?: options.numThreads ?: config.text.generationThreads}",
            "mmap=${options.useMmap ?: config.text.useMmap}",
            "mlock=${options.useMlock ?: config.text.useMlock}",
            "flash=${options.useFlashAttention ?: config.text.useFlashAttention}",
        )
}

internal class TextBackendPolicy(
    private val config: LLMEdgeConfig,
) : BackendPolicy<TextModelOptions> {
    override fun request(options: TextModelOptions) =
        io.aatricks.llmedge.core.runtime.BackendCandidateResolver.Request(
            subsystem = ComputeSubsystem.TEXT,
            allowGpu = options.useVulkan ?: config.text.useVulkan,
            openClAvailable = SmolLM.isOpenClAvailable(),
            vulkanAvailable = SmolLM.isVulkanBackendAvailable(),
        )
}

internal class TextRuntimeLoader(
    private val context: Context,
    private val config: LLMEdgeConfig,
    private val resolver: ModelRepository,
) : RuntimeLoader<ModelSpec, TextModelOptions, ManagedTextModel> {
    override suspend fun load(
        spec: ModelSpec,
        options: TextModelOptions,
        backend: ComputeBackend,
    ): ManagedTextModel {
        val modelFile = resolver.resolve(context, spec)
        val smol = SmolLM(useVulkan = backend == ComputeBackend.VULKAN)
        smol.load(modelFile.absolutePath, options.toInferenceParams(config), preferredBackend = backend)
        return ManagedTextModel(fileSizeBytes = modelFile.length(), model = smol)
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
        descriptor =
            CachedRuntimeDescriptor(
                cache = config.text.cache,
                keyStrategy = TextRuntimeKeyStrategy(config),
                runtimeLoader = TextRuntimeLoader(context, config, resolver),
                activeBackend = { it.model.getActiveBackend() },
                backendPolicy = TextBackendPolicy(config),
            ),
    )
