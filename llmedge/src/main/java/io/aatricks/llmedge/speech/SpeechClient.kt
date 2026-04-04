package io.aatricks.llmedge.speech

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.core.FeatureContext
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.OwnedFeatureClient
import io.aatricks.llmedge.core.createFeatureClientForTesting
import io.aatricks.llmedge.core.createOwnedFeatureClient
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.speech.tts.BarkTTS
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow

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
    featureContext: FeatureContext,
    private val ownedBootstrap: ClientBootstrapContext? = null,
) : OwnedFeatureClient(featureContext, ownedBootstrap) {
    companion object {
        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): SpeechClient = createOwnedFeatureClient(context, scope, config, modelRepository, ::SpeechClient)

        @JvmSynthetic
        internal fun forTesting(
            context: Context,
            scope: LLMEdgeScope,
            config: LLMEdgeConfig,
            resolver: ModelRepository,
            ownedBootstrap: ClientBootstrapContext? = null,
        ): SpeechClient =
            createFeatureClientForTesting(context, scope, config, resolver, ownedBootstrap, ::SpeechClient)
    }

    private val whisperPool = createWhisperRuntimePool(appContext, edgeScope, config, modelRepository)
    private val barkPool = createBarkRuntimePool(appContext, edgeScope, config, modelRepository)
    private val requestExecutor = SpeechRequestExecutor(edgeScope, whisperPool, barkPool)

    /**
     * Preload a Whisper model into the speech cache so later transcription calls avoid the initial
     * model-load cost on the calling path.
     */
    suspend fun prepareSpeechToText(
        model: ModelSpec = config.models.speechToText,
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ) {
        requestExecutor.prepareSpeechToText(model, loadOptions)
    }

    /**
     * Preload a Bark model into the speech cache so later synthesis calls avoid the initial
     * model-load cost on the calling path.
     */
    suspend fun prepareTextToSpeech(
        model: ModelSpec = config.models.textToSpeech,
        loadOptions: BarkLoadOptions = BarkLoadOptions(),
    ) {
        requestExecutor.prepareTextToSpeech(model, loadOptions)
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
        requestExecutor.transcribe(audioSamples, model, params, loadOptions)

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
    ): String? = requestExecutor.detectLanguage(audioSamples, model, loadOptions, nThreads)

    /**
     * Create a reusable real-time transcription session.
     *
     * Call [StreamingTranscriptionSession.close] when the session is no longer needed.
     */
    suspend fun createStreamingSession(
        model: ModelSpec = config.models.speechToText,
        params: Whisper.StreamingParams = Whisper.StreamingParams(),
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ): StreamingTranscriptionSession =
        requestExecutor.createStreamingSession(model, params, loadOptions)

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
    ): BarkTTS.AudioResult = requestExecutor.synthesize(text, model, params, loadOptions)

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
    ): Flow<AudioStreamEvent> = requestExecutor.synthesizeStream(text, model, params, loadOptions)

    override fun close() {
        closeOwned {
            requestExecutor.close()
        }
    }
}
