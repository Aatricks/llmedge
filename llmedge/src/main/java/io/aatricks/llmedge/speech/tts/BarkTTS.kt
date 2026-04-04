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

package io.aatricks.llmedge.speech.tts

import android.content.Context
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeLibraryLoader
import io.aatricks.llmedge.core.NativeLibraryCatalog
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.speech.tts.internal.BarkAudioSupport
import io.aatricks.llmedge.speech.tts.internal.BarkLoaderSupport
import io.aatricks.llmedge.speech.tts.internal.BarkPcmBuffers
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext

/**
 * Kotlin wrapper for bark.cpp providing Text-to-Speech (TTS) functionality.
 *
 * Bark is a transformer-based text-to-speech model that can generate highly realistic speech with
 * various voices and styles. This wrapper enables:
 * - High-quality text-to-speech synthesis
 * - Progress tracking during generation
 * - WAV file output
 *
 * Example usage:
 * ```kotlin
 * val tts = BarkTTS.load(context, "path/to/bark/model")
 *
 * // Generate speech from text
 * val audio = tts.generate("Hello, world!")
 *
 * // Save as WAV file
 * tts.saveAsWav(audio, "output.wav")
 *
 * tts.close()
 * ```
 */
class BarkTTS private constructor(private val handle: Long) : AutoCloseable {
    private val pcmBuffers = BarkPcmBuffers()

    /** Encoding step during synthesis. */
    enum class EncodingStep(val value: Int) {
        SEMANTIC(0),
        COARSE(1),
        FINE(2);

        companion object {
            fun fromInt(value: Int): EncodingStep =
                    entries.firstOrNull { it.value == value } ?: SEMANTIC
        }
    }

    /** Result of audio generation. */
    data class AudioResult(
            /** Raw audio samples as 32-bit float PCM */
            val samples: FloatArray,
            /** Sample rate in Hz (typically 24000 for Bark) */
            val sampleRate: Int,
            /** Duration in seconds */
            val durationSeconds: Float
    ) {
        /** Duration in milliseconds */
        val durationMs: Long
            get() = (durationSeconds * 1000).toLong()

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as AudioResult
            if (!samples.contentEquals(other.samples)) return false
            if (sampleRate != other.sampleRate) return false
            if (durationSeconds != other.durationSeconds) return false
            return true
        }

        override fun hashCode(): Int {
            var result = samples.contentHashCode()
            result = 31 * result + sampleRate
            result = 31 * result + durationSeconds.hashCode()
            return result
        }
    }

    /** Configuration for TTS generation. */
    data class GenerateParams(
            /** Number of threads to use. 0 = auto */
            val nThreads: Int = 0
    )

    /** Callback for generation progress updates. */
    fun interface ProgressCallback {
        /**
         * Called when generation progress is updated.
         * @param step Current encoding step (0=semantic, 1=coarse, 2=fine)
         * @param progress Progress percentage (0-100)
         */
        fun onProgress(step: EncodingStep, progress: Int)
    }

    private var progressCallback: ProgressCallback? = null

    /** Set a callback to receive progress updates during generation. */
    fun setProgressCallback(callback: ProgressCallback?) {
        this.progressCallback = callback
        if (callback != null) {
            nativeSetProgressCallback(
                    handle,
                    object : Any() {
                        @Suppress("unused")
                        fun onProgress(step: Int, progress: Int) {
                            callback.onProgress(EncodingStep.fromInt(step), progress)
                        }
                    }
            )
        } else {
            nativeSetProgressCallback(handle, null)
        }
    }

    /**
     * Generate speech audio from text.
     *
     * @param text Text to synthesize
     * @param params Generation parameters
     * @return AudioResult containing the generated audio samples
     */
    fun generate(text: String, params: GenerateParams = GenerateParams()): AudioResult {
        return BarkAudioSupport.generate(
            text = text,
            params = params,
            nativeGenerate = { input, threads -> nativeGenerate(handle, input, threads) },
            nativeGetSampleRate = { nativeGetSampleRate(handle) },
        )
    }

    /** Generate speech audio and return as a Flow for streaming use cases. */
    fun generateFlow(text: String, params: GenerateParams = GenerateParams()): Flow<AudioResult> =
            flow {
                        val result = generate(text, params)
                        emit(result)
                    }
                    .flowOn(Dispatchers.IO)

    /** Get the sample rate used by the model. */
    fun getSampleRate(): Int = nativeGetSampleRate(handle)

    /** Get model load time in microseconds. */
    fun getLoadTime(): Long = nativeGetLoadTime(handle)

    /** Get model evaluation time in microseconds from last generation. */
    fun getEvalTime(): Long = nativeGetEvalTime(handle)

    /** Reset internal statistics. */
    fun resetStatistics() = nativeResetStatistics(handle)

    /**
     * Save audio samples to a WAV file.
     *
     * @param audio AudioResult from generate()
     * @param filePath Path to save the WAV file
     */
    fun saveAsWav(audio: AudioResult, filePath: String) {
        saveAsWav(audio.samples, audio.sampleRate, filePath)
    }

    /**
     * Save raw audio samples to a WAV file.
     *
     * @param samples Raw audio samples as 32-bit float PCM
     * @param sampleRate Sample rate in Hz
     * @param filePath Path to save the WAV file
     */
    fun saveAsWav(samples: FloatArray, sampleRate: Int, filePath: String) {
        BarkAudioSupport.saveAsWav(samples, sampleRate, filePath, pcmBuffers)
    }

    override fun close() {
        nativeDestroy(handle)
    }

    // Native method declarations
    private external fun nativeCheckBindings(): Boolean
    private external fun nativeCreate(
            modelPath: String,
            seed: Int,
            temp: Float,
            fineTemp: Float,
            verbosity: Int
    ): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeSetProgressCallback(handle: Long, callback: Any?)
    private external fun nativeGenerate(handle: Long, text: String, nThreads: Int): FloatArray?
    private external fun nativeGetSampleRate(handle: Long): Int
    private external fun nativeGetLoadTime(handle: Long): Long
    private external fun nativeGetEvalTime(handle: Long): Long
    private external fun nativeResetStatistics(handle: Long)

    companion object {
        private const val LOG_TAG = "BarkTTS"

        /** Bark default sample rate (24kHz) */
        const val SAMPLE_RATE = 24000

        private fun logD(tag: String, message: String) = AndroidLogAdapter.d(tag, message)

        private fun logI(tag: String, message: String) = AndroidLogAdapter.i(tag, message)

        private fun logW(tag: String, message: String) = AndroidLogAdapter.w(tag, message)

        private fun logE(tag: String, message: String, throwable: Throwable? = null) =
            AndroidLogAdapter.e(tag, message, throwable)

        // Native library loading
        init {
            NativeLibraryLoader.ensureBarkLoaded(
                required = false,
                onDebug = { message -> logD(LOG_TAG, message) },
                onError = { message, throwable -> logE(LOG_TAG, message, throwable) },
            )
        }

        // Dummy instance used to invoke static native methods that are now at the class level.
        private val staticInvoker by lazy { BarkTTS(0L) }

        /** Check if native bindings are available. */
        @JvmStatic
        fun checkBindings(): Boolean = BarkLoaderSupport.checkBindings(staticInvoker::nativeCheckBindings)

        /**
         * Load a Bark model from a file path.
         *
         * The Bark model is a single `ggml_weights.bin` file generated by converting the original
         * Bark weights using bark.cpp's `convert.py` script.
         */
        @JvmStatic
        fun load(
            modelPath: String,
            seed: Int = 0,
            temperature: Float = 0.7f,
            fineTemperature: Float = 0.5f,
            verbosity: Int = 0,
        ): BarkTTS =
            BarkLoaderSupport.loadFromPath(
                modelPath = modelPath,
                create = { absolutePath, resolvedSeed, resolvedTemperature, resolvedFineTemperature, resolvedVerbosity ->
                    NativeCall.requireHandle(
                        NativeCall.binding(
                            NativeLibraryCatalog.BARK,
                            "Bark JNI bindings are unavailable.",
                        ) {
                            staticInvoker.nativeCreate(
                                absolutePath,
                                resolvedSeed,
                                resolvedTemperature,
                                resolvedFineTemperature,
                                resolvedVerbosity,
                            )
                        },
                        absolutePath,
                        "The native Bark loader returned an invalid handle.",
                    )
                },
                seed = seed,
                temperature = temperature,
                fineTemperature = fineTemperature,
                verbosity = verbosity,
            )

        /**
         * Load a Bark model with Android Context support.
         *
         * @param context Android context
         * @param modelPath Path to the model file (can be relative to cache dir)
         * @param seed Random seed for reproducibility
         * @param temperature Sampling temperature for text/coarse encoders
         * @param fineTemperature Sampling temperature for fine encoder
         * @param verbosity Verbosity level
         * @return BarkTTS instance
         */
        @JvmStatic
        suspend fun load(
                context: Context,
                modelPath: String,
                seed: Int = 0,
                temperature: Float = 0.7f,
                fineTemperature: Float = 0.5f,
                verbosity: Int = 0
        ): BarkTTS =
            BarkLoaderSupport.loadWithContext(
                context = context,
                modelPath = modelPath,
                loadFromPath = ::load,
                seed = seed,
                temperature = temperature,
                fineTemperature = fineTemperature,
                verbosity = verbosity,
            )

        /**
         * Download and load a Bark model from Hugging Face Hub.
         *
         * Note: Bark models require conversion to GGML format before use. The model file should be
         * a pre-converted ggml weights file (e.g., bark-small_weights-f16.bin).
         *
         * @param context Android context for caching
         * @param modelId Hugging Face model ID (default: "Green-Sky/bark-ggml")
         * @param filename The model filename (default: "bark-small_weights-f16.bin")
         * @param seed Random seed for reproducibility
         * @param temperature Sampling temperature for text/coarse encoders
         * @param fineTemperature Sampling temperature for fine encoder
         * @param verbosity Verbosity level
         * @param token Optional Hugging Face API token for private models
         * @return BarkTTS instance
         */
        @JvmStatic
        suspend fun loadFromHuggingFace(
                context: Context,
                modelId: String = "Green-Sky/bark-ggml",
                filename: String = "bark-small_weights-f16.bin",
                seed: Int = 0,
                temperature: Float = 0.7f,
                fineTemperature: Float = 0.5f,
                verbosity: Int = 0,
                token: String? = null
        ): BarkTTS =
            BarkLoaderSupport.loadFromHuggingFace(
                context = context,
                modelId = modelId,
                filename = filename,
                seed = seed,
                temperature = temperature,
                fineTemperature = fineTemperature,
                verbosity = verbosity,
                token = token,
                loadFromPath = ::load,
            )

        internal fun createFromHandleForRuntime(handle: Long): BarkTTS = BarkTTS(handle)

    }
}
