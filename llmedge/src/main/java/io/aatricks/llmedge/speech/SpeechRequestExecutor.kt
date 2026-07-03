package io.aatricks.llmedge.speech

import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.speech.tts.BarkTTS
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch

internal class SpeechRequestExecutor(
    private val scope: LLMEdgeScope,
    private val whisperPool: RuntimePool<ModelSpec, WhisperLoadOptions, ManagedWhisperModel>,
    private val barkPool: RuntimePool<ModelSpec, BarkLoadOptions, ManagedBarkModel>,
) {
    suspend fun prepareSpeechToText(
        model: ModelSpec,
        loadOptions: WhisperLoadOptions,
    ) {
        whisperPool.prepare(model, loadOptions)
    }

    suspend fun prepareTextToSpeech(
        model: ModelSpec,
        loadOptions: BarkLoadOptions,
    ) {
        barkPool.prepare(model, loadOptions)
    }

    suspend fun transcribe(
        audioSamples: FloatArray,
        model: ModelSpec,
        params: Whisper.TranscribeParams,
        loadOptions: WhisperLoadOptions,
    ): List<Whisper.TranscriptionSegment> =
        withWhisperRuntime(model, loadOptions) { runtime ->
            runtime.whisper.transcribe(audioSamples, params)
        }

    suspend fun detectLanguage(
        audioSamples: FloatArray,
        model: ModelSpec,
        loadOptions: WhisperLoadOptions,
        nThreads: Int,
    ): String? =
        withWhisperRuntime(model, loadOptions) { runtime ->
            runtime.whisper.detectLanguage(audioSamples, nThreads)
        }

    suspend fun createStreamingSession(
        model: ModelSpec,
        params: Whisper.StreamingParams,
        loadOptions: WhisperLoadOptions,
    ): StreamingTranscriptionSession {
        // Lease (pin) the runtime for the whole session: the transcriber keeps calling
        // into it long after this function returns, so plain acquire would let cache
        // pressure close it mid-stream.
        val lease = whisperPool.acquireLeased(model, loadOptions)
        val transcriber =
            try {
                lease.runtime.whisper.createStreamingTranscriber(params)
            } catch (t: Throwable) {
                lease.close()
                throw t
            }
        return scope.resources.register(StreamingTranscriptionSession(transcriber, lease))
    }

    suspend fun synthesize(
        text: String,
        model: ModelSpec,
        params: BarkTTS.GenerateParams,
        loadOptions: BarkLoadOptions,
    ): BarkTTS.AudioResult =
        withBarkRuntime(model, loadOptions) { runtime ->
            runtime.bark.generate(text, params)
        }

    fun synthesizeStream(
        text: String,
        model: ModelSpec,
        params: BarkTTS.GenerateParams,
        loadOptions: BarkLoadOptions,
    ): Flow<AudioStreamEvent> = callbackFlow {
        trySend(AudioStreamEvent.Started)
        val job =
            scope.coroutineScope.launch {
                barkPool.withExclusiveRuntimeRetry(
                    spec = model,
                    options = loadOptions,
                    dispatcher = scope.inferenceDispatcher,
                ) { runtime, _ ->
                    runtime.bark.setProgressCallback { step, progress ->
                        trySend(AudioStreamEvent.Progress(step, progress))
                    }
                    try {
                        val result = runtime.bark.generate(text, params)
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
        }
    }

    fun close() {
        try {
            barkPool.close()
        } finally {
            whisperPool.close()
        }
    }

    private suspend fun <T> withWhisperRuntime(
        model: ModelSpec,
        options: WhisperLoadOptions,
        block: suspend (ManagedWhisperModel) -> T,
    ): T =
        whisperPool.withExclusiveRuntimeRetry(
            spec = model,
            options = options,
            dispatcher = scope.inferenceDispatcher,
        ) { runtime, _ ->
            block(runtime)
        }

    private suspend fun <T> withBarkRuntime(
        model: ModelSpec,
        options: BarkLoadOptions,
        block: suspend (ManagedBarkModel) -> T,
    ): T =
        barkPool.withExclusiveRuntimeRetry(
            spec = model,
            options = options,
            dispatcher = scope.inferenceDispatcher,
        ) { runtime, _ ->
            block(runtime)
        }
}
