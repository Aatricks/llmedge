package io.aatricks.llmedge.speech

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.speech.tts.BarkTTS
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
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

class SpeechClient internal constructor(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val resolver: ModelResolver,
) : AutoCloseable {
    private val whisperPool = createWhisperRuntimePool(context, scope, config, resolver)
    private val barkPool = createBarkRuntimePool(context, scope, config, resolver)

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
    ): List<Whisper.TranscriptionSegment> =
        withWhisperRuntime(model, loadOptions) { runtime ->
            runtime.whisper.transcribe(audioSamples, params)
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
    ): String? =
        withWhisperRuntime(model, loadOptions) { runtime ->
            runtime.whisper.detectLanguage(audioSamples, nThreads)
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
        val transcriber =
            withWhisperRuntime(model, loadOptions) { runtime ->
                runtime.whisper.createStreamingTranscriber(params)
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
    ): BarkTTS.AudioResult =
        withBarkRuntime(model, loadOptions) { runtime ->
            runtime.bark.generate(text, params)
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
    ): ManagedWhisperModel = whisperPool.acquire(model, options)

    private fun recordWhisperBackendFailureIfNeeded(
        model: ModelSpec,
        options: WhisperLoadOptions,
        runtime: ManagedWhisperModel,
        error: InferenceFailedException,
    ): Boolean = whisperPool.recordBackendFailureIfNeeded(model, options, runtime, error)

    private suspend fun acquireBark(
        model: ModelSpec,
        options: BarkLoadOptions,
    ): ManagedBarkModel = barkPool.acquire(model, options)

    private suspend fun <T> withWhisperRuntime(
        model: ModelSpec,
        options: WhisperLoadOptions,
        block: suspend (ManagedWhisperModel) -> T,
    ): T {
        val runtime = acquireWhisper(model, options)
        return try {
            runtime.mutex.withLock {
                withContext(scope.inferenceDispatcher) {
                    block(runtime)
                }
            }
        } catch (error: InferenceFailedException) {
            if (recordWhisperBackendFailureIfNeeded(model, options, runtime, error)) {
                return withWhisperRuntime(model, options, block)
            }
            throw error
        }
    }

    private suspend fun <T> withBarkRuntime(
        model: ModelSpec,
        options: BarkLoadOptions,
        block: suspend (ManagedBarkModel) -> T,
    ): T {
        val runtime = acquireBark(model, options)
        return runtime.mutex.withLock {
            withContext(scope.inferenceDispatcher) {
                block(runtime)
            }
        }
    }

    override fun close() {
        barkPool.close()
        whisperPool.close()
    }
}
