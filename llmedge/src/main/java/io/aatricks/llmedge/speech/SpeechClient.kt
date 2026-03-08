package io.aatricks.llmedge.speech

import android.content.Context
import io.aatricks.llmedge.BarkTTS
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.ModelCache
import io.aatricks.llmedge.Whisper
import io.aatricks.llmedge.core.LLMEdgeScope
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
) : AutoCloseable {
    val mutex: Mutex = Mutex()

    override fun close() {
        whisper.close()
    }
}

private class ManagedBarkModel(
    val fileSizeBytes: Long,
    val bark: BarkTTS,
) : AutoCloseable {
    val mutex: Mutex = Mutex()

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
        ModelCache<ManagedWhisperModel>(
            maxCacheSize = config.speechCacheSize,
            maxMemoryMB = config.speechCacheMemoryMb,
            closeScope = scope.coroutineScope,
        )
    private val barkCache =
        ModelCache<ManagedBarkModel>(
            maxCacheSize = config.speechCacheSize,
            maxMemoryMB = config.speechCacheMemoryMb,
            closeScope = scope.coroutineScope,
        )
    private val whisperLoadMutex = Mutex()
    private val barkLoadMutex = Mutex()

    suspend fun transcribe(
        audioSamples: FloatArray,
        model: ModelSpec = config.models.speechToText,
        params: Whisper.TranscribeParams = Whisper.TranscribeParams(),
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ): List<Whisper.TranscriptionSegment> {
        val runtime = acquireWhisper(model, loadOptions)
        return runtime.mutex.withLock {
            withContext(scope.inferenceDispatcher) {
                runtime.whisper.transcribe(audioSamples, params)
            }
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
        return runtime.mutex.withLock {
            withContext(scope.inferenceDispatcher) {
                runtime.whisper.detectLanguage(audioSamples, nThreads)
            }
        }
    }

    suspend fun createStreamingSession(
        model: ModelSpec = config.models.speechToText,
        params: Whisper.StreamingParams = Whisper.StreamingParams(),
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ): StreamingTranscriptionSession {
        val runtime = acquireWhisper(model, loadOptions)
        val transcriber = runtime.mutex.withLock { runtime.whisper.createStreamingTranscriber(params) }
        return scope.resources.register(StreamingTranscriptionSession(transcriber))
    }

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
        val key = listOf(model.cacheKey, options.useGpu, options.flashAttention, options.gpuDevice).joinToString("|")
        whisperCache.get(key)?.let { return it }
        return whisperLoadMutex.withLock {
            whisperCache.get(key)?.let { return@withLock it }
            val file = resolver.resolve(context, model)
            val runtime = ManagedWhisperModel(
                fileSizeBytes = file.length(),
                whisper = Whisper.load(file.absolutePath, options.useGpu, options.flashAttention, options.gpuDevice),
            )
            whisperCache.put(key, runtime, runtime.fileSizeBytes)
            runtime
        }
    }

    private suspend fun acquireBark(
        model: ModelSpec,
        options: BarkLoadOptions,
    ): ManagedBarkModel {
        val key =
            listOf(
                model.cacheKey,
                options.seed,
                options.temperature,
                options.fineTemperature,
                options.verbosity,
            ).joinToString("|")
        barkCache.get(key)?.let { return it }
        return barkLoadMutex.withLock {
            barkCache.get(key)?.let { return@withLock it }
            val file = resolver.resolve(context, model)
            val runtime = ManagedBarkModel(
                fileSizeBytes = file.length(),
                bark = BarkTTS.load(
                    modelPath = file.absolutePath,
                    seed = options.seed,
                    temperature = options.temperature,
                    fineTemperature = options.fineTemperature,
                    verbosity = options.verbosity,
                ),
            )
            barkCache.put(key, runtime, runtime.fileSizeBytes)
            runtime
        }
    }

    override fun close() {
        barkCache.clear()
        whisperCache.clear()
    }
}
