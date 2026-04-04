package io.aatricks.llmedge.speech

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.ManagedRuntimeBase
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
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

internal fun createWhisperRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    resolver: ModelRepository,
): RuntimePool<ModelSpec, WhisperLoadOptions, ManagedWhisperModel> =
    createCachedRuntimePool(
        context = context,
        scope = scope,
        cacheConfig = config.speech.cache,
        cacheKeyPrefix = { spec, options ->
            RuntimeCacheKeyBuilder.prefix(
                spec.cacheKey,
                "gpu=${options.useGpu}",
                "flash=${options.flashAttention}",
                "gpuDevice=${options.gpuDevice}",
            )
        },
        loadRuntime = { spec, options, backend ->
            val file = resolver.resolve(context, spec)
            ManagedWhisperModel(
                fileSizeBytes = file.length(),
                whisper = Whisper.load(file.absolutePath, backend, options.flashAttention, options.gpuDevice),
            )
        },
        activeBackend = { it.whisper.activeBackend },
        candidateRequest = { options ->
            BackendCandidateResolver.Request(
                subsystem = ComputeSubsystem.WHISPER,
                allowGpu = options.useGpu,
                openClAvailable = Whisper.isOpenClAvailable(),
                vulkanAvailable = Whisper.isVulkanBackendAvailable(),
            )
        },
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
        cacheConfig = config.speech.cache,
        cacheKeyPrefix = { spec, options ->
            RuntimeCacheKeyBuilder.prefix(
                spec.cacheKey,
                "seed=${options.seed}",
                "temperature=${options.temperature}",
                "fineTemperature=${options.fineTemperature}",
                "verbosity=${options.verbosity}",
            )
        },
        loadRuntime = { spec, options, _ ->
            val file = resolver.resolve(context, spec)
            ManagedBarkModel(
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
        },
        activeBackend = { ComputeBackend.CPU },
        candidateRequest = {
            BackendCandidateResolver.Request(
                subsystem = null,
                allowGpu = false,
                openClAvailable = false,
                vulkanAvailable = false,
            )
        },
    )
