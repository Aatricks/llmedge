package io.aatricks.llmedge.text

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ModelCacheFactory
import io.aatricks.llmedge.core.runtime.BackendPolicy
import io.aatricks.llmedge.core.runtime.ManagedRuntime
import io.aatricks.llmedge.core.runtime.RuntimeKeyStrategy
import io.aatricks.llmedge.core.runtime.RuntimeLoader
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean

internal class ManagedTextModel(
    val fileSizeBytes: Long,
    val model: SmolLM,
) : ManagedRuntime {
    override val mutex: Mutex = Mutex()
    private val closed = AtomicBoolean(false)

    private fun estimatedNativeMemoryBytes(): Long =
        maxOf(
            model.getEstimatedNativeMemoryBytes().takeIf { it > 0L } ?: 0L,
            fileSizeBytes + model.getEstimatedStateMemoryBytes().coerceAtLeast(0L),
            fileSizeBytes,
        )

    override fun estimatedSizeBytes(): Long = estimatedNativeMemoryBytes()

    fun ensureOpen() {
        check(!closed.get()) { "Text runtime has been closed" }
    }

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
    private val resolver: ModelResolver,
) : RuntimeLoader<ModelSpec, TextModelOptions, ManagedTextModel> {
    override suspend fun load(
        spec: ModelSpec,
        options: TextModelOptions,
    ): ManagedTextModel {
        val modelFile = resolver.resolve(context, spec)
        val smol = SmolLM(useVulkan = options.useVulkan ?: config.text.useVulkan)
        smol.load(modelFile.absolutePath, options.toInferenceParams(config))
        return ManagedTextModel(fileSizeBytes = modelFile.length(), model = smol)
    }
}

internal fun createTextRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    resolver: ModelResolver,
): RuntimePool<ModelSpec, TextModelOptions, ManagedTextModel> =
    RuntimePool(
        cache =
            ModelCacheFactory.create(
                context = context,
                scope = scope,
                maxCacheSize = config.text.cache.maxEntries,
                maxMemoryMB = config.text.cache.maxMemoryMb,
            ),
        keyStrategy = TextRuntimeKeyStrategy(config),
        runtimeLoader = TextRuntimeLoader(context, config, resolver),
        activeBackend = { it.model.getActiveBackend() },
        backendPolicy = TextBackendPolicy(config),
    )
