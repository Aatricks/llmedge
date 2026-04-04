/*
 * Copyright (C) 2024 LLMEdge Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.aatricks.llmedge.speech.stt

import android.content.Context
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.core.NativeBridgeProvider
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeLibraryLoader
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.speech.stt.internal.WhisperCompanionSupport
import io.aatricks.llmedge.speech.stt.internal.WhisperInferenceOperations
import io.aatricks.llmedge.speech.stt.internal.WhisperStreamingState
import io.aatricks.llmedge.speech.stt.internal.WhisperSubtitleSupport
import io.aatricks.llmedge.runtime.ComputeBackend
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext

/**
 * Kotlin wrapper for whisper.cpp providing Speech-to-Text (STT) functionality.
 *
 * Whisper is an automatic speech recognition (ASR) model that can transcribe and translate audio in
 * multiple languages. This wrapper enables:
 * - Real-time transcription
 * - Translation between languages
 * - Subtitle generation with timestamps
 * - Language detection
 *
 * Example usage:
 * ```kotlin
 * val whisper = Whisper.load(context, "path/to/ggml-base.bin")
 *
 * // Transcribe audio samples (16kHz mono PCM float32)
 * val segments = whisper.transcribe(audioSamples)
 * segments.forEach { segment ->
 *     println("[${segment.startTimeMs}ms - ${segment.endTimeMs}ms] ${segment.text}")
 * }
 *
 * whisper.close()
 * ```
 */
class Whisper internal constructor(
    private val handle: Long,
    internal val activeBackend: ComputeBackend = ComputeBackend.CPU,
) : AutoCloseable {

    /** Represents a transcribed segment with timing information. */
    data class TranscriptionSegment(
            val index: Int,
            val startTime: Long, // Start time in whisper time units (centiseconds)
            val endTime: Long, // End time in whisper time units (centiseconds)
            val text: String
    ) {
        /** Start time in milliseconds */
        val startTimeMs: Long
            get() = startTime * 10

        /** End time in milliseconds */
        val endTimeMs: Long
            get() = endTime * 10

        /** Duration in milliseconds */
        val durationMs: Long
            get() = endTimeMs - startTimeMs

        /** Format as SRT subtitle entry */
        fun toSrtEntry(): String = WhisperSubtitleSupport.toSrtEntry(this)

        /** Format as VTT subtitle entry */
        fun toVttEntry(): String = WhisperSubtitleSupport.toVttEntry(this)
    }

    /** Configuration for transcription. */
    data class TranscribeParams(
            /** Number of threads to use. 0 = auto */
            val nThreads: Int = 0,
            /** Translate to English instead of transcribing */
            val translate: Boolean = false,
            /** Target language code (e.g., "en", "es", "fr"). null = auto-detect */
            val language: String? = null,
            /** Force language detection even if language is specified */
            val detectLanguage: Boolean = false,
            /** Enable token-level timestamps for more precise timing */
            val tokenTimestamps: Boolean = false,
            /** Maximum segment length in characters. 0 = no limit */
            val maxLen: Int = 0,
            /** Split on word boundaries when using maxLen */
            val splitOnWord: Boolean = false,
            /** Sampling temperature (0.0 = greedy, higher = more random) */
            val temperature: Float = 0.0f,
            /** Beam size for beam search (1 = greedy decoding) */
            val beamSize: Int = 1,
            /** Suppress blank tokens at the beginning of segments */
            val suppressBlank: Boolean = true,
            /** Print progress to console */
            val printProgress: Boolean = false
    )

    /** Callback for transcription progress updates. */
    fun interface ProgressCallback {
        fun onProgress(progress: Int)
    }

    /** Callback for new transcription segments (for real-time streaming). */
    fun interface SegmentCallback {
        fun onNewSegment(index: Int, startTime: Long, endTime: Long, text: String)
    }

    private var progressCallback: ProgressCallback? = null
    private var segmentCallback: SegmentCallback? = null

    // Internal bridge interface for testing
    internal interface NativeBridge {
        fun transcribe(
                handle: Long,
                samples: FloatArray,
                params: TranscribeParams,
                progressCallback: ProgressCallback?,
                segmentCallback: SegmentCallback?
        ): Array<TranscriptionSegment>?

        fun detectLanguage(handle: Long, samples: FloatArray, nThreads: Int): Int
        fun getFullText(handle: Long): String
        fun close(handle: Long)
    }

    /** Set a callback to receive progress updates during transcription. */
    fun setProgressCallback(callback: ProgressCallback?) {
        this.progressCallback = callback
        if (callback != null) {
            nativeSetProgressCallback(
                    handle,
                    object : Any() {
                        @Suppress("unused")
                        fun onProgress(progress: Int) {
                            callback.onProgress(progress)
                        }
                    }
            )
        } else {
            nativeSetProgressCallback(handle, null)
        }
    }

    /**
     * Set a callback to receive new segments in real-time during transcription. This is useful for
     * streaming transcription results.
     */
    fun setSegmentCallback(callback: SegmentCallback?) {
        this.segmentCallback = callback
        if (callback != null) {
            nativeSetSegmentCallback(
                    handle,
                    object : Any() {
                        @Suppress("unused")
                        fun onNewSegment(index: Int, startTime: Long, endTime: Long, text: String) {
                            callback.onNewSegment(index, startTime, endTime, text)
                        }
                    }
            )
        } else {
            nativeSetSegmentCallback(handle, null)
        }
    }

    /**
     * Transcribe audio samples to text.
     *
     * @param samples Audio samples as 32-bit float PCM at 16kHz mono
     * @param params Transcription parameters
     * @return List of transcription segments with timing information
     */
    fun transcribe(
            samples: FloatArray,
            params: TranscribeParams = TranscribeParams()
    ): List<TranscriptionSegment> =
        WhisperInferenceOperations.transcribe(
            samples = samples,
            params = params,
        ) { pcm, threads, translate, language, detectLanguage, tokenTimestamps, maxLen, splitOnWord, temperature, beamSize, suppressBlank, printProgress ->
            nativeTranscribe(
                handle,
                pcm,
                threads,
                translate,
                language,
                detectLanguage,
                tokenTimestamps,
                maxLen,
                splitOnWord,
                temperature,
                beamSize,
                suppressBlank,
                printProgress,
            )
        }

    /** Transcribe audio and return results as a Flow for streaming use cases. */
    fun transcribeFlow(
            samples: FloatArray,
            params: TranscribeParams = TranscribeParams()
    ): Flow<TranscriptionSegment> =
            flow {
                        val segments = transcribe(samples, params)
                        segments.forEach { emit(it) }
                    }
                    .flowOn(Dispatchers.IO)

    /**
     * Detect the language of the audio.
     *
     * @param samples Audio samples as 32-bit float PCM at 16kHz mono
     * @param nThreads Number of threads to use. 0 = auto
     * @return Language code (e.g., "en", "es", "fr") or null if detection fails
     */
    fun detectLanguage(samples: FloatArray, nThreads: Int = 0): String? {
        return WhisperInferenceOperations.detectLanguage(
            samples = samples,
            nThreads = nThreads,
            detectLanguageNative = { pcm, threads -> nativeDetectLanguage(handle, pcm, threads, 0) },
            resolveLanguageString = ::getLanguageString,
        )
    }

    /** Get the full transcribed text from the last transcription. */
    fun getFullText(): String = nativeGetFullText(handle)

    /** Check if the loaded model supports multiple languages. */
    fun isMultilingual(): Boolean = nativeIsMultilingual(handle)

    /** Get the model type (e.g., "tiny", "base", "small", "medium", "large"). */
    fun getModelType(): String = nativeGetModelType(handle)

    /** Reset internal timing statistics. */
    fun resetTimings() = nativeResetTimings(handle)

    /** Print timing statistics to the log. */
    fun printTimings() = nativePrintTimings(handle)

    /** Generate SRT subtitle content from transcription segments. */
    fun generateSrt(segments: List<TranscriptionSegment>): String =
        WhisperSubtitleSupport.generateSrt(segments)

    /** Generate WebVTT subtitle content from transcription segments. */
    fun generateVtt(segments: List<TranscriptionSegment>): String =
        WhisperSubtitleSupport.generateVtt(segments)

    /**
     * Create a streaming transcriber for real-time audio transcription.
     *
     * The streaming transcriber uses a sliding window approach:
     * - Collects audio in chunks of `stepMs` milliseconds
     * - Transcribes a window of `lengthMs` milliseconds at a time
     * - Keeps `keepMs` of audio from the previous window for context
     *
     * This enables real-time transcription as audio is being recorded.
     *
     * @param params Streaming transcription parameters
     * @return StreamingTranscriber instance
     */
    fun createStreamingTranscriber(
            params: StreamingParams = StreamingParams()
    ): StreamingTranscriber {
        return StreamingTranscriber(this, params)
    }

    /** Parameters for streaming transcription. */
    data class StreamingParams(
            /**
             * Duration in milliseconds of each step (how often transcription runs). Default: 3000ms
             */
            val stepMs: Int = 3000,
            /** Length of the transcription window in milliseconds. Default: 10000ms */
            val lengthMs: Int = 10000,
            /** Audio from previous window to keep for context. Default: 200ms */
            val keepMs: Int = 200,
            /** Translate to English instead of transcribing */
            val translate: Boolean = false,
            /** Target language code. null = auto-detect */
            val language: String? = null,
            /** Number of threads. 0 = auto */
            val nThreads: Int = 0,
            /**
             * Voice Activity Detection threshold (0.0-1.0). Higher = more aggressive silence
             * detection
             */
            val vadThreshold: Float = 0.6f,
            /** High-pass frequency cutoff for VAD in Hz */
            val vadFreqThreshold: Float = 100.0f,
            /** Enable VAD to only transcribe when speech is detected */
            val useVad: Boolean = true
    )

    /**
     * Real-time streaming transcriber using a sliding window approach.
     *
     * Inspired by whisper.cpp's stream example, this class buffers incoming audio and processes it
     * in overlapping windows to provide near real-time transcription.
     *
     * Usage:
     * ```kotlin
     * val transcriber = whisper.createStreamingTranscriber()
     * transcriber.start().collect { segment ->
     *     println("Transcribed: ${segment.text}")
     * }
     *
     * // Feed audio samples as they become available (from microphone, etc.)
     * transcriber.feedAudio(audioChunk)
     *
     * // When done
     * transcriber.stop()
     * ```
     */
    class StreamingTranscriber
    internal constructor(private val whisper: Whisper, private val params: StreamingParams) {
        private val state = WhisperStreamingState(whisper, params)

        /**
         * Feed audio samples to the streaming transcriber.
         *
         * Audio should be:
         * - 16kHz sample rate
         * - Mono channel
         * - 32-bit float PCM (-1.0 to 1.0)
         *
         * @param samples Audio samples to add to the buffer
         */
        suspend fun feedAudio(samples: FloatArray) = state.feedAudio(samples)

        /** Get the current audio buffer size in milliseconds. */
        fun getBufferedAudioMs(): Int = state.getBufferedAudioMs()

        /** Check if enough audio is buffered for processing. */
        fun hasEnoughAudio(): Boolean = state.hasEnoughAudio()

        /** Clear the audio buffer. */
        suspend fun clearBuffer() = state.clearBuffer()

        /** Pause audio collection (transcription continues with buffered audio). */
        fun pause() = state.pause()

        /** Resume audio collection. */
        fun resume() = state.resume()

        /** Stop the streaming transcription. */
        fun stop() = state.stop()

        /**
         * Start streaming transcription, returning a Flow of transcription segments.
         *
         * This is a cold flow - transcription starts when collection begins and stops when
         * collection is cancelled or stop() is called.
         *
         * @return Flow emitting TranscriptionSegment as they are transcribed
         */
        fun start(): Flow<TranscriptionSegment> = state.start()

        /**
         * Process a single chunk of audio immediately.
         *
         * This is useful when you want more control over when transcription happens, rather than
         * using the continuous start() flow.
         *
         * @return List of transcription segments from this chunk
         */
        suspend fun processNextChunk(): List<TranscriptionSegment> = state.processNextChunk()
    }

    override fun close() {
        nativeDestroy(handle)
    }

    // Native method declarations
    private external fun nativeCheckBindings(): Boolean
    private external fun nativeGetVersion(): String
    private external fun nativeGetSystemInfo(): String
    private external fun nativeGetMaxLanguageId(): Int
    private external fun nativeGetLanguageId(lang: String): Int
    private external fun nativeGetLanguageString(langId: Int): String
    private external fun nativeIsOpenClAvailable(): Boolean
    private external fun nativeIsVulkanAvailable(): Boolean
    private external fun nativeCreate(
            modelPath: String,
            backendId: Int,
            flashAttn: Boolean,
            gpuDevice: Int
    ): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeIsMultilingual(handle: Long): Boolean
    private external fun nativeGetModelType(handle: Long): String
    private external fun nativeSetProgressCallback(handle: Long, callback: Any?)
    private external fun nativeSetSegmentCallback(handle: Long, callback: Any?)
    private external fun nativeTranscribe(
            handle: Long,
            samples: FloatArray,
            nThreads: Int,
            translate: Boolean,
            language: String?,
            detectLanguage: Boolean,
            tokenTimestamps: Boolean,
            maxLen: Int,
            splitOnWord: Boolean,
            temperature: Float,
            beamSize: Int,
            suppressBlank: Boolean,
            printProgress: Boolean
    ): Array<TranscriptionSegment>?
    private external fun nativeDetectLanguage(
            handle: Long,
            samples: FloatArray,
            nThreads: Int,
            offsetMs: Int
    ): Int
    private external fun nativeGetFullText(handle: Long): String
    private external fun nativeResetTimings(handle: Long)
    private external fun nativePrintTimings(handle: Long)

    internal fun supportCheckBindings(): Boolean = nativeCheckBindings()

    internal fun supportGetVersion(): String = nativeGetVersion()

    internal fun supportGetSystemInfo(): String = nativeGetSystemInfo()

    internal fun supportGetMaxLanguageId(): Int = nativeGetMaxLanguageId()

    internal fun supportGetLanguageId(lang: String): Int = nativeGetLanguageId(lang)

    internal fun supportGetLanguageString(langId: Int): String = nativeGetLanguageString(langId)

    internal fun supportIsOpenClAvailable(): Boolean = nativeIsOpenClAvailable()

    internal fun supportIsVulkanAvailable(): Boolean = nativeIsVulkanAvailable()

    companion object {
        private const val LOG_TAG = "Whisper"

        /** Whisper expects audio at 16kHz sample rate */
        const val SAMPLE_RATE = 16000

        /** Whisper processes audio in 30-second chunks */
        const val CHUNK_SIZE_SECONDS = 30

        private fun logD(tag: String, message: String) = AndroidLogAdapter.d(tag, message)

        private fun logI(tag: String, message: String) = AndroidLogAdapter.i(tag, message)

        private fun logW(tag: String, message: String) = AndroidLogAdapter.w(tag, message)

        private fun logE(tag: String, message: String, throwable: Throwable? = null) =
            AndroidLogAdapter.e(tag, message, throwable)

        // Native library loading - similar to SmolLM
        init {
            NativeLibraryLoader.ensureWhisperLoaded(
                required = false,
                onDebug = { message -> logD(LOG_TAG, message) },
                onError = { message, throwable -> logE(LOG_TAG, message, throwable) },
            )
        }

        internal interface LoadBridge {
            fun create(
                modelPath: String,
                backend: ComputeBackend,
                flashAttn: Boolean,
                gpuDevice: Int,
            ): Long
        }

        // Dummy instance used to invoke static native methods that are now at the class level.
        internal val staticInvoker by lazy { Whisper(0L, ComputeBackend.CPU) }
        internal val loadBridgeProvider =
            NativeBridgeProvider<Unit, LoadBridge> { _ ->
                object : LoadBridge {
                    override fun create(
                        modelPath: String,
                        backend: ComputeBackend,
                        flashAttn: Boolean,
                        gpuDevice: Int,
                    ): Long =
                        staticInvoker.nativeCreate(
                            modelPath,
                            backend.id,
                            flashAttn,
                            gpuDevice,
                        )
                }
            }

        internal var openClAvailabilityOverrideForTests: Boolean? = null
        internal var vulkanAvailabilityOverrideForTests: Boolean? = null

        internal fun overrideLoadBridgeForTests(provider: () -> LoadBridge) {
            loadBridgeProvider.override { _ -> provider() }
        }

        internal fun resetLoadBridgeForTests() {
            loadBridgeProvider.reset()
        }

        internal fun overrideBackendAvailabilityForTests(
            openClAvailable: Boolean? = null,
            vulkanAvailable: Boolean? = null,
        ) {
            openClAvailabilityOverrideForTests = openClAvailable
            vulkanAvailabilityOverrideForTests = vulkanAvailable
        }

        internal fun resetBackendAvailabilityForTests() {
            openClAvailabilityOverrideForTests = null
            vulkanAvailabilityOverrideForTests = null
        }

        /** Check if native bindings are available. */
        @JvmStatic
        fun checkBindings(): Boolean = WhisperCompanionSupport.checkBindings(staticInvoker)

        /** Get the whisper.cpp version string. */
        @JvmStatic fun getVersion(): String = WhisperCompanionSupport.getVersion(staticInvoker)

        /** Get system information string. */
        @JvmStatic fun getSystemInfo(): String = WhisperCompanionSupport.getSystemInfo(staticInvoker)

        /** Get the maximum language ID supported. */
        @JvmStatic fun getMaxLanguageId(): Int = WhisperCompanionSupport.getMaxLanguageId(staticInvoker)

        /**
         * Get the language ID for a language code or name.
         *
         * @param lang Language code (e.g., "en") or name (e.g., "english")
         * @return Language ID or -1 if not found
         */
        @JvmStatic fun getLanguageId(lang: String): Int = WhisperCompanionSupport.getLanguageId(staticInvoker, lang)

        /**
         * Get the language code for a language ID.
         *
         * @param langId Language ID
         * @return Language code (e.g., "en") or empty string if not found
         */
        @JvmStatic fun getLanguageString(langId: Int): String =
            WhisperCompanionSupport.getLanguageString(staticInvoker, langId)

        @JvmStatic
        fun isOpenClAvailable(): Boolean =
            WhisperCompanionSupport.isOpenClAvailable(staticInvoker, openClAvailabilityOverrideForTests)

        @JvmStatic
        fun isVulkanBackendAvailable(): Boolean =
            WhisperCompanionSupport.isVulkanBackendAvailable(staticInvoker, vulkanAvailabilityOverrideForTests)

        /**
         * Load a Whisper model from a file path.
         *
         * @param modelPath Path to the GGML Whisper model file
         * @param useGpu Enable GPU acceleration (if available)
         * @param flashAttn Enable flash attention optimization
         * @param gpuDevice GPU device index (for multi-GPU systems)
         * @return Whisper instance
         */
        @JvmStatic
        fun load(
                modelPath: String,
                useGpu: Boolean = false,
                flashAttn: Boolean = true,
                gpuDevice: Int = 0
        ): Whisper =
            WhisperCompanionSupport.load(
                modelPath = modelPath,
                useGpu = useGpu,
                flashAttn = flashAttn,
                gpuDevice = gpuDevice,
                staticInvoker = staticInvoker,
                createHandle = { path, backend, useFlashAttn, device ->
                    loadBridgeProvider.create(Unit).create(path, backend, useFlashAttn, device)
                },
                openClAvailabilityOverride = openClAvailabilityOverrideForTests,
                vulkanAvailabilityOverride = vulkanAvailabilityOverrideForTests,
            ) { backend ->
                logW(LOG_TAG, "Failed to load Whisper on $backend; retrying with the next backend")
            }

        internal fun load(
            modelPath: String,
            backend: ComputeBackend,
            flashAttn: Boolean = true,
            gpuDevice: Int = 0,
        ): Whisper =
            WhisperCompanionSupport.loadOnBackend(
                modelPath = modelPath,
                backend = backend,
                flashAttn = flashAttn,
                gpuDevice = gpuDevice,
            ) { path, chosenBackend, useFlashAttn, device ->
                loadBridgeProvider.create(Unit).create(path, chosenBackend, useFlashAttn, device)
            }

        /**
         * Load a Whisper model with Android Context support. This allows loading models from app
         * assets or cache directories.
         *
         * @param context Android context
         * @param modelPath Path to the model file (can be relative to cache dir)
         * @param useGpu Enable GPU acceleration
         * @param flashAttn Enable flash attention optimization
         * @param gpuDevice GPU device index
         * @return Whisper instance
         */
        @JvmStatic
        suspend fun load(
                context: Context,
                modelPath: String,
                useGpu: Boolean = false,
                flashAttn: Boolean = true,
                gpuDevice: Int = 0
        ): Whisper =
            WhisperCompanionSupport.load(
                context = context,
                modelPath = modelPath,
                useGpu = useGpu,
                flashAttn = flashAttn,
                gpuDevice = gpuDevice,
                loadFromPath = ::load,
            )

        /**
         * Download and load a Whisper model from Hugging Face Hub.
         *
         * @param context Android context for caching
         * @param modelId Hugging Face model ID (e.g., "ggerganov/whisper.cpp")
         * @param modelFile Specific model file name (e.g., "ggml-base.bin")
         * @param useGpu Enable GPU acceleration
         * @param flashAttn Enable flash attention optimization
         * @param gpuDevice GPU device index
         * @param token Optional Hugging Face API token for private models
         * @return Whisper instance
         */
        @JvmStatic
        suspend fun loadFromHuggingFace(
                context: Context,
                modelId: String = "ggerganov/whisper.cpp",
                modelFile: String = "ggml-base.bin",
                useGpu: Boolean = false,
                flashAttn: Boolean = true,
                gpuDevice: Int = 0,
                token: String? = null
        ): Whisper =
            WhisperCompanionSupport.loadFromHuggingFace(
                context = context,
                modelId = modelId,
                modelFile = modelFile,
                useGpu = useGpu,
                flashAttn = flashAttn,
                gpuDevice = gpuDevice,
                token = token,
                loadFromPath = ::load,
            )
    }
}
