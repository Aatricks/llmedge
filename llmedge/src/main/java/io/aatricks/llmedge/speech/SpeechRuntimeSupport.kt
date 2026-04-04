package io.aatricks.llmedge.speech

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.BackendPolicy
import io.aatricks.llmedge.core.runtime.CachedRuntimeDescriptor
import io.aatricks.llmedge.core.runtime.ManagedRuntimeBase
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.core.runtime.RuntimeKeyStrategy
import io.aatricks.llmedge.core.runtime.RuntimeLoader
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.createCachedRuntimePool
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.speech.tts.BarkTTS

internal class ManagedWhisperModel(
    val fileSizeBytes: Long,
    val whisper: Whisper,
) : ManagedRuntimeBase() {
    override fun estimatedSizeBytes(): Long = fileSizeBytes

    override fun close() {
        closeOnce(whisper::close)
    }
}

internal class ManagedBarkModel(
    val fileSizeBytes: Long,
    val bark: BarkTTS,
) : ManagedRuntimeBase() {
    override fun estimatedSizeBytes(): Long = fileSizeBytes

    override fun close() {
        closeOnce(bark::close)
    }
}

internal class WhisperRuntimeKeyStrategy : RuntimeKeyStrategy<ModelSpec, WhisperLoadOptions> {
    override fun prefix(
        spec: ModelSpec,
        options: WhisperLoadOptions,
    ): String =
        RuntimeCacheKeyBuilder.prefix(
            spec.cacheKey,
            "gpu=${options.useGpu}",
            "flash=${options.flashAttention}",
            "gpuDevice=${options.gpuDevice}",
        )
}

internal class BarkRuntimeKeyStrategy : RuntimeKeyStrategy<ModelSpec, BarkLoadOptions> {
    override fun prefix(
        spec: ModelSpec,
        options: BarkLoadOptions,
    ): String =
        RuntimeCacheKeyBuilder.prefix(
            spec.cacheKey,
            "seed=${options.seed}",
            "temperature=${options.temperature}",
            "fineTemperature=${options.fineTemperature}",
            "verbosity=${options.verbosity}",
        )
}

internal class WhisperBackendPolicy : BackendPolicy<WhisperLoadOptions> {
    override fun request(options: WhisperLoadOptions) =
        BackendCandidateResolver.Request(
            subsystem = ComputeSubsystem.WHISPER,
            allowGpu = options.useGpu,
            openClAvailable = Whisper.isOpenClAvailable(),
            vulkanAvailable = Whisper.isVulkanBackendAvailable(),
        )
}

internal class BarkBackendPolicy : BackendPolicy<BarkLoadOptions> {
    override fun request(options: BarkLoadOptions) =
        BackendCandidateResolver.Request(
            subsystem = null,
            allowGpu = false,
            openClAvailable = false,
            vulkanAvailable = false,
        )
}

internal class WhisperRuntimeLoader(
    private val context: Context,
    private val resolver: ModelRepository,
) : RuntimeLoader<ModelSpec, WhisperLoadOptions, ManagedWhisperModel> {
    override suspend fun load(
        spec: ModelSpec,
        options: WhisperLoadOptions,
        backend: ComputeBackend,
    ): ManagedWhisperModel {
        val file = resolver.resolve(context, spec)
        return ManagedWhisperModel(
            fileSizeBytes = file.length(),
            whisper = Whisper.load(file.absolutePath, backend, options.flashAttention, options.gpuDevice),
        )
    }
}

internal class BarkRuntimeLoader(
    private val context: Context,
    private val resolver: ModelRepository,
) : RuntimeLoader<ModelSpec, BarkLoadOptions, ManagedBarkModel> {
    override suspend fun load(
        spec: ModelSpec,
        options: BarkLoadOptions,
        backend: ComputeBackend,
    ): ManagedBarkModel {
        val file = resolver.resolve(context, spec)
        return ManagedBarkModel(
            fileSizeBytes = file.length(),
            bark =
                BarkTTS.load(
                    modelPath = file.absolutePath,
                    seed = options.seed,
                    temperature = options.temperature,
                    fineTemperature = options.fineTemperature,
                    verbosity = options.verbosity,
                ),
        )
    }
}

internal fun createWhisperRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    resolver: ModelRepository,
): RuntimePool<ModelSpec, WhisperLoadOptions, ManagedWhisperModel> =
    createCachedRuntimePool(
        context = context,
        scope = scope,
        descriptor =
            CachedRuntimeDescriptor(
                cache = config.speech.cache,
                keyStrategy = WhisperRuntimeKeyStrategy(),
                runtimeLoader = WhisperRuntimeLoader(context, resolver),
                activeBackend = { it.whisper.activeBackend },
                backendPolicy = WhisperBackendPolicy(),
            ),
    )

internal fun createBarkRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    resolver: ModelRepository,
): RuntimePool<ModelSpec, BarkLoadOptions, ManagedBarkModel> =
    createCachedRuntimePool(
        context = context,
        scope = scope,
        descriptor =
            CachedRuntimeDescriptor(
                cache = config.speech.cache,
                keyStrategy = BarkRuntimeKeyStrategy(),
                runtimeLoader = BarkRuntimeLoader(context, resolver),
                activeBackend = { ComputeBackend.CPU },
                backendPolicy = BarkBackendPolicy(),
            ),
    )
