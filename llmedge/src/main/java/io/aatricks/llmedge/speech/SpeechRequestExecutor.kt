package io.aatricks.llmedge.speech

import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.executeWithRuntimeRetry
import io.aatricks.llmedge.core.runtime.runExclusive
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
        whisperPool.acquire(model, loadOptions)
    }

    suspend fun prepareTextToSpeech(
        model: ModelSpec,
        loadOptions: BarkLoadOptions,
    ) {
        barkPool.acquire(model, loadOptions)
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
        val transcriber =
            withWhisperRuntime(model, loadOptions) { runtime ->
                runtime.whisper.createStreamingTranscriber(params)
            }
        return scope.resources.register(StreamingTranscriptionSession(transcriber))
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
        val runtime = barkPool.acquire(model, loadOptions)
        trySend(AudioStreamEvent.Started)
        val job =
            scope.coroutineScope.launch {
                runtime.runExclusive(scope.inferenceDispatcher) {
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
            runtime.bark.setProgressCallback(null)
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
        whisperPool.executeWithRuntimeRetry(
            spec = model,
            options = options,
        ) { execution ->
            execution.runtime.runExclusive(scope.inferenceDispatcher, block)
        }

    private suspend fun <T> withBarkRuntime(
        model: ModelSpec,
        options: BarkLoadOptions,
        block: suspend (ManagedBarkModel) -> T,
    ): T {
        val runtime = barkPool.acquire(model, options)
        return runtime.runExclusive(scope.inferenceDispatcher, block)
    }
}
