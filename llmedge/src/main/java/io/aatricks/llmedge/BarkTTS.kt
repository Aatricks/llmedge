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

package io.aatricks.llmedge

import android.content.Context
import android.os.Build
import android.util.Log
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
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

    /** Reusable ByteBuffer for WAV PCM conversion to avoid per-call allocation */
    private var pcmBuffer: ByteBuffer? = null

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
        require(text.isNotEmpty()) { "Text cannot be empty" }

        val effectiveThreads =
                if (params.nThreads <= 0) {
                    Runtime.getRuntime().availableProcessors().coerceAtMost(8)
                } else {
                    params.nThreads
                }

        val samples =
                nativeGenerate(handle, text, effectiveThreads)
                        ?: throw RuntimeException("Failed to generate audio")

        val sampleRate = nativeGetSampleRate(handle)
        val durationSeconds = samples.size.toFloat() / sampleRate

        return AudioResult(samples, sampleRate, durationSeconds)
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
        val file = File(filePath)
        file.parentFile?.mkdirs()

        FileOutputStream(file).use { fos ->
            val wavHeader = createWavHeader(samples.size, sampleRate)
            fos.write(wavHeader)

            // Convert float samples to 16-bit PCM
            val requiredBytes = samples.size * 2
            var buffer = pcmBuffer
            if (buffer == null || buffer.capacity() < requiredBytes) {
                buffer = ByteBuffer.allocate(requiredBytes).order(ByteOrder.LITTLE_ENDIAN)
                pcmBuffer = buffer
            } else {
                buffer.clear()
                buffer.order(ByteOrder.LITTLE_ENDIAN)
            }
            for (sample in samples) {
                // Clamp and convert to 16-bit
                val clamped = sample.coerceIn(-1.0f, 1.0f)
                val pcm16 = (clamped * 32767.0f).toInt().toShort()
                buffer.putShort(pcm16)
            }
            fos.write(buffer.array())
        }
    }

    private fun createWavHeader(numSamples: Int, sampleRate: Int): ByteArray {
        val byteRate = sampleRate * 2 // 16-bit mono
        val dataSize = numSamples * 2
        val fileSize = 36 + dataSize

        val buffer = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header
        buffer.put(WAV_RIFF)
        buffer.putInt(fileSize)
        buffer.put(WAV_WAVE)

        // fmt subchunk
        buffer.put(WAV_FMT)
        buffer.putInt(16) // Subchunk1Size (16 for PCM)
        buffer.putShort(1) // AudioFormat (1 for PCM)
        buffer.putShort(1) // NumChannels (1 for mono)
        buffer.putInt(sampleRate) // SampleRate
        buffer.putInt(byteRate) // ByteRate
        buffer.putShort(2) // BlockAlign (2 for 16-bit mono)
        buffer.putShort(16) // BitsPerSample

        // data subchunk
        buffer.put(WAV_DATA)
        buffer.putInt(dataSize)

        return buffer.array()
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

        // Pre-computed WAV header tag bytes to avoid repeated toByteArray() allocations
        private val WAV_RIFF = "RIFF".toByteArray()
        private val WAV_WAVE = "WAVE".toByteArray()
        private val WAV_FMT = "fmt ".toByteArray()
        private val WAV_DATA = "data".toByteArray()

        private val isAndroidLogAvailable: Boolean =
                try {
                    Class.forName("android.util.Log")
                    true
                } catch (_: Throwable) {
                    false
                }

        // Cache reflected Log methods to avoid Class.forName + getMethod on every call
        private val cachedLogD: java.lang.reflect.Method? by lazy {
            try { Class.forName("android.util.Log").getMethod("d", String::class.java, String::class.java) } catch (_: Throwable) { null }
        }
        private val cachedLogI: java.lang.reflect.Method? by lazy {
            try { Class.forName("android.util.Log").getMethod("i", String::class.java, String::class.java) } catch (_: Throwable) { null }
        }
        private val cachedLogW: java.lang.reflect.Method? by lazy {
            try { Class.forName("android.util.Log").getMethod("w", String::class.java, String::class.java) } catch (_: Throwable) { null }
        }
        private val cachedLogE: java.lang.reflect.Method? by lazy {
            try { Class.forName("android.util.Log").getMethod("e", String::class.java, String::class.java, Throwable::class.java) } catch (_: Throwable) { null }
        }

        private fun logD(tag: String, message: String) {
            val method = cachedLogD
            if (method != null) {
                try { method.invoke(null, tag, message) } catch (_: Throwable) { println("D/$tag: $message") }
            } else {
                println("D/$tag: $message")
            }
        }

        private fun logI(tag: String, message: String) {
            val method = cachedLogI
            if (method != null) {
                try { method.invoke(null, tag, message) } catch (_: Throwable) { println("I/$tag: $message") }
            } else {
                println("I/$tag: $message")
            }
        }

        private fun logW(tag: String, message: String) {
            val method = cachedLogW
            if (method != null) {
                try { method.invoke(null, tag, message) } catch (_: Throwable) { println("W/$tag: $message") }
            } else {
                println("W/$tag: $message")
            }
        }

        private fun logE(tag: String, message: String, throwable: Throwable? = null) {
            val method = cachedLogE
            if (method != null) {
                try { method.invoke(null, tag, message, throwable) } catch (_: Throwable) {
                    System.err.println("E/$tag: $message")
                    throwable?.printStackTrace()
                }
            } else {
                System.err.println("E/$tag: $message")
                throwable?.printStackTrace()
            }
        }

        // Native library loading
        init {
            val disableNativeLoad = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")
            if (disableNativeLoad) {
                println("[BarkTTS] Native library load disabled via llmedge.disableNativeLoad=true")
            } else {
                try {
                    // Load bark_jni library - same name on all platforms
                    logD(LOG_TAG, "Loading libbark_jni.so")
                    System.loadLibrary("bark_jni")
                } catch (e: UnsatisfiedLinkError) {
                    logE(LOG_TAG, "Failed to load bark native library: ${e.message}")
                }
            }
        }

                // Dummy instance used to invoke static native methods that are now at the class level.

                private val staticInvoker by lazy { BarkTTS(0L) }

        

                private fun supportsArm64V8a(): Boolean {

                    return Build.SUPPORTED_64_BIT_ABIS.any { it == "arm64-v8a" }

                }

        

                /** Check if native bindings are available. */

                @JvmStatic

                fun checkBindings(): Boolean {

                    return try {

                        staticInvoker.nativeCheckBindings()

                    } catch (e: UnsatisfiedLinkError) {

                        false

                    }

                }

        

                /**

                 * Load a Bark model from a file path.

                 *

                 * The bark model is a single ggml_weights.bin file generated by converting the original

                 * Bark model using the convert.py script from bark.cpp.

                 *

                 * @param modelPath Path to the ggml_weights.bin model file

                 * @param seed Random seed for reproducibility (0 = random)

                 * @param temperature Sampling temperature for text/coarse encoders (default 0.7)

                 * @param fineTemperature Sampling temperature for fine encoder (default 0.5)

                 * @param verbosity Verbosity level (0=low, 1=medium, 2=high)

                 * @return BarkTTS instance

                 */

                @JvmStatic

                fun load(

                        modelPath: String,

                        seed: Int = 0,

                        temperature: Float = 0.7f,

                        fineTemperature: Float = 0.5f,

                        verbosity: Int = 0

                ): BarkTTS {

                    val file = File(modelPath)

                    if (!file.exists()) {

                        throw FileNotFoundException("Model file not found: $modelPath")

                    }

        

                    val handle = staticInvoker.nativeCreate(modelPath, seed, temperature, fineTemperature, verbosity)

                    if (handle == 0L) {

                        throw RuntimeException("Failed to load Bark model from: $modelPath")

                    }

                    return BarkTTS(handle)

                }

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
                withContext(Dispatchers.IO) {
                    val file = File(modelPath)
                    val actualPath =
                            if (file.exists()) {
                                modelPath
                            } else {
                                // Try to find in cache directory
                                val cacheFile = File(context.cacheDir, modelPath)
                                if (cacheFile.exists()) {
                                    cacheFile.absolutePath
                                } else {
                                    // Try to find in files directory
                                    val filesFile = File(context.filesDir, modelPath)
                                    if (filesFile.exists()) {
                                        filesFile.absolutePath
                                    } else {
                                        throw FileNotFoundException(
                                                "Model file not found: $modelPath"
                                        )
                                    }
                                }
                            }

                    load(actualPath, seed, temperature, fineTemperature, verbosity)
                }

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
                withContext(Dispatchers.IO) {
                    // Download the single ggml_weights.bin file
                    val result =
                            HuggingFaceHub.ensureModelOnDisk(
                                    context = context,
                                    modelId = modelId,
                                    filename = filename,
                                    token = token
                            )

                    load(result.file.absolutePath, seed, temperature, fineTemperature, verbosity)
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
    }
}
