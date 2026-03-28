package io.aatricks.llmedge.speech

import android.content.Context
import io.aatricks.llmedge.speech.tts.BarkTTS
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.core.ModelCacheFactory
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.BackendPolicy
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.ManagedRuntime
import io.aatricks.llmedge.core.runtime.RuntimeKeyStrategy
import io.aatricks.llmedge.core.runtime.RuntimeLoader
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.RuntimeCacheKeyBuilder
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class WhisperLoadOptions(
    val useGpu: Boolean = false,
    val flashAttention: Boolean = true,
    val gpuDevice: Int = 0,
)

data class BarkLoadOptions(
    val seed: Int = 0,
    val temperature: Float = 0.7f,
    val fineTemperature: Float = 0.5f,
    val verbosity: Int = 0,
)

class StreamingTranscriptionSession internal constructor(
    private val transcriber: Whisper.StreamingTranscriber,
) : AutoCloseable {
    fun events(): Flow<Whisper.TranscriptionSegment> = transcriber.start()

    suspend fun feedAudio(samples: FloatArray) {
        transcriber.feedAudio(samples)
    }

    fun stop() {
        transcriber.stop()
    }

    override fun close() {
        stop()
    }
}

private class ManagedWhisperModel(
    val fileSizeBytes: Long,
    val whisper: Whisper,
) : ManagedRuntime {
    override val mutex: Mutex = Mutex()

    override fun estimatedSizeBytes(): Long = fileSizeBytes

    override fun close() {
        whisper.close()
    }
}

private class ManagedBarkModel(
    val fileSizeBytes: Long,
    val bark: BarkTTS,
) : ManagedRuntime {
    override val mutex: Mutex = Mutex()

    override fun estimatedSizeBytes(): Long = fileSizeBytes

    override fun close() {
        bark.close()
    }
}

class SpeechClient internal constructor(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val resolver: ModelResolver,
) : AutoCloseable {
    private val whisperCache =
        ModelCacheFactory.create<ManagedWhisperModel>(
            context = context,
            scope = scope,
            maxCacheSize = config.speechCacheSize,
            maxMemoryMB = config.speechCacheMemoryMb,
        )
    private val barkCache =
        ModelCacheFactory.create<ManagedBarkModel>(
            context = context,
            scope = scope,
            maxCacheSize = config.speechCacheSize,
            maxMemoryMB = config.speechCacheMemoryMb,
        )
    private val whisperPool =
        RuntimePool<ModelSpec, WhisperLoadOptions, ManagedWhisperModel>(
            cache = whisperCache,
            keyStrategy =
                RuntimeKeyStrategy { model, options ->
                    RuntimeCacheKeyBuilder.prefix(
                        model.cacheKey,
                        options.useGpu,
                        options.flashAttention,
                        options.gpuDevice,
                    )
                },
            runtimeLoader =
                RuntimeLoader { model, options ->
                    val file = resolver.resolve(context, model)
                    ManagedWhisperModel(
                        fileSizeBytes = file.length(),
                        whisper = Whisper.load(file.absolutePath, options.useGpu, options.flashAttention, options.gpuDevice),
                    )
                },
            activeBackend = { it.whisper.activeBackend },
            backendPolicy =
                BackendPolicy { options ->
                    BackendCandidateResolver.Request(
                        subsystem = ComputeSubsystem.WHISPER,
                        allowGpu = options.useGpu,
                        openClAvailable = Whisper.isOpenClAvailable(),
                        vulkanAvailable = Whisper.isVulkanBackendAvailable(),
                    )
                },
        )
    private val barkPool =
        RuntimePool<ModelSpec, BarkLoadOptions, ManagedBarkModel>(
            cache = barkCache,
            keyStrategy =
                RuntimeKeyStrategy { model, options ->
                    RuntimeCacheKeyBuilder.prefix(
                        model.cacheKey,
                        options.seed,
                        options.temperature,
                        options.fineTemperature,
                        options.verbosity,
                    )
                },
            runtimeLoader =
                RuntimeLoader { model, options ->
                    val file = resolver.resolve(context, model)
                    ManagedBarkModel(
                        fileSizeBytes = file.length(),
                        bark = BarkTTS.load(
                            modelPath = file.absolutePath,
                            seed = options.seed,
                            temperature = options.temperature,
                            fineTemperature = options.fineTemperature,
                            verbosity = options.verbosity,
                        ),
                    )
                },
            activeBackend = { io.aatricks.llmedge.runtime.ComputeBackend.CPU },
            backendPolicy =
                BackendPolicy {
                    BackendCandidateResolver.Request(
                        subsystem = null,
                        allowGpu = false,
                        openClAvailable = false,
                        vulkanAvailable = false,
                    )
                },
        )

    /**
     * Preload a Whisper model into the speech cache so later transcription calls avoid the initial
     * model-load cost on the calling path.
     */
    suspend fun prepareSpeechToText(
        model: ModelSpec = config.models.speechToText,
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ) {
        whisperPool.acquire(model, loadOptions)
    }

    /**
     * Preload a Bark model into the speech cache so later synthesis calls avoid the initial
     * model-load cost on the calling path.
     */
    suspend fun prepareTextToSpeech(
        model: ModelSpec = config.models.textToSpeech,
        loadOptions: BarkLoadOptions = BarkLoadOptions(),
    ) {
        barkPool.acquire(model, loadOptions)
    }

    /**
     * Transcribe an audio buffer into timestamped speech segments.
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution, loading, or native
     * inference fails.
     */
    suspend fun transcribe(
        audioSamples: FloatArray,
        model: ModelSpec = config.models.speechToText,
        params: Whisper.TranscribeParams = Whisper.TranscribeParams(),
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ): List<Whisper.TranscriptionSegment> {
        val runtime = acquireWhisper(model, loadOptions)
        return try {
            runtime.mutex.withLock {
                withContext(scope.inferenceDispatcher) {
                    runtime.whisper.transcribe(audioSamples, params)
                }
            }
        } catch (error: io.aatricks.llmedge.core.InferenceFailedException) {
            if (recordWhisperBackendFailureIfNeeded(model, loadOptions, runtime, error)) {
                return transcribe(audioSamples, model, params, loadOptions)
            }
            throw error
        }
    }

    suspend fun transcribeToText(
        audioSamples: FloatArray,
        model: ModelSpec = config.models.speechToText,
        params: Whisper.TranscribeParams = Whisper.TranscribeParams(),
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ): String = transcribe(audioSamples, model, params, loadOptions).joinToString(" ") { it.text.trim() }

    suspend fun detectLanguage(
        audioSamples: FloatArray,
        model: ModelSpec = config.models.speechToText,
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
        nThreads: Int = 0,
    ): String? {
        val runtime = acquireWhisper(model, loadOptions)
        return try {
            runtime.mutex.withLock {
                withContext(scope.inferenceDispatcher) {
                    runtime.whisper.detectLanguage(audioSamples, nThreads)
                }
            }
        } catch (error: io.aatricks.llmedge.core.InferenceFailedException) {
            if (recordWhisperBackendFailureIfNeeded(model, loadOptions, runtime, error)) {
                return detectLanguage(audioSamples, model, loadOptions, nThreads)
            }
            throw error
        }
    }

    /**
     * Create a reusable real-time transcription session.
     *
     * Call [StreamingTranscriptionSession.close] when the session is no longer needed.
     */
    suspend fun createStreamingSession(
        model: ModelSpec = config.models.speechToText,
        params: Whisper.StreamingParams = Whisper.StreamingParams(),
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ): StreamingTranscriptionSession {
        val runtime = acquireWhisper(model, loadOptions)
        val transcriber =
            try {
                runtime.mutex.withLock { runtime.whisper.createStreamingTranscriber(params) }
            } catch (error: io.aatricks.llmedge.core.InferenceFailedException) {
                if (recordWhisperBackendFailureIfNeeded(model, loadOptions, runtime, error)) {
                    return createStreamingSession(model, params, loadOptions)
                }
                throw error
            }
        return scope.resources.register(StreamingTranscriptionSession(transcriber))
    }

    /**
     * Synthesize speech from text and return the full audio result.
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution, loading, or native
     * inference fails.
     */
    suspend fun synthesize(
        text: String,
        model: ModelSpec = config.models.textToSpeech,
        params: BarkTTS.GenerateParams = BarkTTS.GenerateParams(),
        loadOptions: BarkLoadOptions = BarkLoadOptions(),
    ): BarkTTS.AudioResult {
        val runtime = acquireBark(model, loadOptions)
        return runtime.mutex.withLock {
            withContext(scope.inferenceDispatcher) {
                runtime.bark.generate(text, params)
            }
        }
    }

    /**
     * Stream Bark synthesis progress followed by the final audio result.
     *
     * Cancel the returned flow collection to stop listening for the active synthesis result.
     */
    fun synthesizeStream(
        text: String,
        model: ModelSpec = config.models.textToSpeech,
        params: BarkTTS.GenerateParams = BarkTTS.GenerateParams(),
        loadOptions: BarkLoadOptions = BarkLoadOptions(),
    ): Flow<AudioStreamEvent> = callbackFlow {
        val runtime = acquireBark(model, loadOptions)
        trySend(AudioStreamEvent.Started)
        val job = scope.coroutineScope.launch {
            runtime.mutex.withLock {
                runtime.bark.setProgressCallback { step, progress ->
                    trySend(AudioStreamEvent.Progress(step, progress))
                }
                try {
                    val result = withContext(scope.inferenceDispatcher) { runtime.bark.generate(text, params) }
                    trySend(AudioStreamEvent.Result(result))
                    trySend(AudioStreamEvent.Completed)
                    close()
                } catch (t: Throwable) {
                    close(t)
                } finally {
                    runtime.bark.setProgressCallback(null)
                }
            }
        }
        awaitClose {
            job.cancel()
            runtime.bark.setProgressCallback(null)
        }
    }

    private suspend fun acquireWhisper(
        model: ModelSpec,
        options: WhisperLoadOptions,
    ): ManagedWhisperModel {
        return whisperPool.acquire(model, options)
    }

    private fun recordWhisperBackendFailureIfNeeded(
        model: ModelSpec,
        options: WhisperLoadOptions,
        runtime: ManagedWhisperModel,
        error: io.aatricks.llmedge.core.InferenceFailedException,
    ): Boolean {
        val blacklisted = whisperPool.recordBackendFailureIfNeeded(model, options, runtime, error)
        if (!blacklisted) {
            return false
        }
        return true
    }

    private suspend fun acquireBark(
        model: ModelSpec,
        options: BarkLoadOptions,
    ): ManagedBarkModel {
        return barkPool.acquire(model, options)
    }

    override fun close() {
        barkPool.close()
        whisperPool.close()
    }
}
