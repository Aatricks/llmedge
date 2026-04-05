package io.aatricks.llmedge.speech

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.core.FeatureContext
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.OwnedFeatureClient
import io.aatricks.llmedge.core.featureClientFactory
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

data class WhisperRuntimeRequest(
    val gpuEnabled: Boolean = false,
    val flashAttention: Boolean = true,
    val gpuDevice: Int = 0,
) {
    internal fun toLoadOptions(): WhisperLoadOptions =
        WhisperLoadOptions(
            useGpu = gpuEnabled,
            flashAttention = flashAttention,
            gpuDevice = gpuDevice,
        )
}

data class BarkRuntimeRequest(
    val seed: Int = 0,
    val temperature: Float = 0.7f,
    val fineTemperature: Float = 0.5f,
    val verbosity: Int = 0,
) {
    internal fun toLoadOptions(): BarkLoadOptions =
        BarkLoadOptions(
            seed = seed,
            temperature = temperature,
            fineTemperature = fineTemperature,
            verbosity = verbosity,
        )
}

data class SpeechToTextPrepareRequest(
    val model: ModelSpec,
    val runtime: WhisperRuntimeRequest = WhisperRuntimeRequest(),
)

data class SpeechSynthesisPrepareRequest(
    val model: ModelSpec,
    val runtime: BarkRuntimeRequest = BarkRuntimeRequest(),
)

data class SpeechToTextRequest(
    val audioSamples: FloatArray,
    val model: ModelSpec,
    val params: Whisper.TranscribeParams = Whisper.TranscribeParams(),
    val runtime: WhisperRuntimeRequest = WhisperRuntimeRequest(),
)

data class SpeechLanguageDetectionRequest(
    val audioSamples: FloatArray,
    val model: ModelSpec,
    val promptThreads: Int = 0,
    val runtime: WhisperRuntimeRequest = WhisperRuntimeRequest(),
)

data class StreamingTranscriptionRequest(
    val model: ModelSpec,
    val params: Whisper.StreamingParams = Whisper.StreamingParams(),
    val runtime: WhisperRuntimeRequest = WhisperRuntimeRequest(),
)

data class SpeechSynthesisRequest(
    val text: String,
    val model: ModelSpec,
    val params: BarkTTS.GenerateParams = BarkTTS.GenerateParams(),
    val runtime: BarkRuntimeRequest = BarkRuntimeRequest(),
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
        private val FACTORY = featureClientFactory(::SpeechClient)

        @Deprecated(
            message = "Prefer LLMEdge.create(...).speech in new app code. This factory remains available for advanced construction and tests.",
        )
        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): SpeechClient = FACTORY.create(context, scope, config, modelRepository, Unit)

        @JvmSynthetic
        internal fun forTesting(
            context: Context,
            scope: LLMEdgeScope,
            config: LLMEdgeConfig,
            resolver: ModelRepository,
            ownedBootstrap: ClientBootstrapContext? = null,
        ): SpeechClient =
            FACTORY.forTesting(context, scope, config, resolver, Unit, ownedBootstrap)
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

    suspend fun prepareSpeechToText(
        request: SpeechToTextPrepareRequest,
    ) {
        requestExecutor.prepareSpeechToText(request.model, request.runtime.toLoadOptions())
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

    suspend fun prepareTextToSpeech(
        request: SpeechSynthesisPrepareRequest,
    ) {
        requestExecutor.prepareTextToSpeech(request.model, request.runtime.toLoadOptions())
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

    suspend fun transcribe(
        request: SpeechToTextRequest,
    ): List<Whisper.TranscriptionSegment> =
        requestExecutor.transcribe(
            request.audioSamples,
            request.model,
            request.params,
            request.runtime.toLoadOptions(),
        )

    suspend fun transcribeToText(
        audioSamples: FloatArray,
        model: ModelSpec = config.models.speechToText,
        params: Whisper.TranscribeParams = Whisper.TranscribeParams(),
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
    ): String = transcribe(audioSamples, model, params, loadOptions).joinToString(" ") { it.text.trim() }

    suspend fun transcribeToText(
        request: SpeechToTextRequest,
    ): String = transcribe(request).joinToString(" ") { it.text.trim() }

    suspend fun detectLanguage(
        audioSamples: FloatArray,
        model: ModelSpec = config.models.speechToText,
        loadOptions: WhisperLoadOptions = WhisperLoadOptions(),
        nThreads: Int = 0,
    ): String? = requestExecutor.detectLanguage(audioSamples, model, loadOptions, nThreads)

    suspend fun detectLanguage(
        request: SpeechLanguageDetectionRequest,
    ): String? =
        requestExecutor.detectLanguage(
            request.audioSamples,
            request.model,
            request.runtime.toLoadOptions(),
            request.promptThreads,
        )

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

    suspend fun createStreamingSession(
        request: StreamingTranscriptionRequest,
    ): StreamingTranscriptionSession =
        requestExecutor.createStreamingSession(
            request.model,
            request.params,
            request.runtime.toLoadOptions(),
        )

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

    suspend fun synthesize(
        request: SpeechSynthesisRequest,
    ): BarkTTS.AudioResult =
        requestExecutor.synthesize(
            request.text,
            request.model,
            request.params,
            request.runtime.toLoadOptions(),
        )

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

    fun synthesizeStream(
        request: SpeechSynthesisRequest,
    ): Flow<AudioStreamEvent> =
        requestExecutor.synthesizeStream(
            request.text,
            request.model,
            request.params,
            request.runtime.toLoadOptions(),
        )

    override fun close() {
        closeOwned {
            requestExecutor.close()
        }
    }
}
