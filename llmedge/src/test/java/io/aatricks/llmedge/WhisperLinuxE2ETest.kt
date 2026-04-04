package io.aatricks.llmedge

import android.content.Context
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import java.io.File
import kotlin.math.sin
import io.aatricks.llmedge.speech.stt.Whisper

/**
 * Linux-host end-to-end test for Whisper speech transcription using a real native library
 * (libwhisper_jni.so) and a local model path.
 *
 * Requirements to run:
 * - Build the native whisper library for Linux and place as llmedge/build/native/linux-x86_64/libwhisper_jni.so
 *   (Use scripts/build_native_linux.sh whisper).
 * - Provide a test model via environment variable LLMEDGE_TEST_WHISPER_MODEL_PATH
 *   (path to a ggml whisper model file). If not set, the test will be skipped.
 *
 * This test exercises the full transcription pipeline from audio samples to text segments.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class WhisperLinuxE2ETest {

    private val LIB_PATH_ENV = "LLMEDGE_BUILD_WHISPER_LIB_PATH"
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_WHISPER_MODEL_PATH"

    /**
     * Generate a simple test audio signal - a sine wave that simulates speech.
     * Real transcription tests should use actual audio files.
     */
    private fun generateTestAudio(durationSeconds: Float = 2.0f): FloatArray {
        val sampleRate = Whisper.SAMPLE_RATE
        val numSamples = (sampleRate * durationSeconds).toInt()
        val samples = FloatArray(numSamples)

        // Generate a simple sine wave (not real speech, but tests the pipeline)
        val frequency = 440.0f  // A4 note
        for (i in 0 until numSamples) {
            val t = i.toFloat() / sampleRate
            samples[i] = (sin(2.0 * Math.PI * frequency * t) * 0.5f).toFloat()
        }

        return samples
    }

    @Test
    fun `desktop end-to-end whisper transcription`() = runBlocking {
        // Skip test if model path is not provided
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        println("[WhisperLinuxE2ETest] modelPath=$modelPath")
        Assume.assumeTrue("No test model specified in $MODEL_PATH_ENV", !modelPath.isNullOrBlank())

        // Check that native library path is properly set and library file exists
        val libPath = System.getenv(LIB_PATH_ENV)
            ?: System.getProperty(LIB_PATH_ENV)
            ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libwhisper_jni.so"

        val libFile = File(libPath)
        println("[WhisperLinuxE2ETest] libPath=$libPath libExists=${libFile.exists()}")
        println("[WhisperLinuxE2ETest] java.library.path=${System.getProperty("java.library.path")}")
        Assume.assumeTrue("Native library not found at $libPath", libFile.exists())

        // Verify native library loading is enabled
        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        println("[WhisperLinuxE2ETest] llmedge.disableNativeLoad=$disableNativeLoad")
        Assume.assumeTrue(
            "Native loading is disabled. Run with LLMEDGE_BUILD_WHISPER_LIB_PATH env var set.",
            disableNativeLoad != "true"
        )

        // Check if model file exists
        val modelFile = File(modelPath)
        Assume.assumeTrue("Model file not found at $modelPath", modelFile.exists())

        println("[WhisperLinuxE2ETest] Loading Whisper model...")
        val whisper = try {
            Whisper.load(
                modelPath = modelPath,
                useGpu = false,
                flashAttn = true
            )
        } catch (e: Exception) {
            println("[WhisperLinuxE2ETest] Failed to load model: ${e.message}")
            e.printStackTrace()
            throw e
        }

        println("[WhisperLinuxE2ETest] Model loaded, model type: ${whisper.getModelType()}")
        println("[WhisperLinuxE2ETest] Multilingual: ${whisper.isMultilingual()}")

        // Generate test audio
        val testAudio = generateTestAudio(2.0f)
        println("[WhisperLinuxE2ETest] Generated ${testAudio.size} audio samples (${testAudio.size / Whisper.SAMPLE_RATE.toFloat()}s)")

        // Test transcription
        val startTime = System.currentTimeMillis()
        val segments = try {
            whisper.transcribe(
                samples = testAudio,
                params = Whisper.TranscribeParams(
                    nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4),
                    language = "en",
                    printProgress = true
                )
            )
        } catch (e: Exception) {
            println("[WhisperLinuxE2ETest] Transcription failed: ${e.message}")
            e.printStackTrace()
            throw e
        } finally {
            whisper.close()
        }

        val elapsed = System.currentTimeMillis() - startTime
        println("[WhisperLinuxE2ETest] Transcription completed in ${elapsed}ms, got ${segments.size} segments")

        // The test audio is just a sine wave, so we don't expect meaningful text
        // But the pipeline should complete without errors
        // Real tests with actual speech audio would verify the transcription content

        segments.forEachIndexed { index, segment ->
            println("[WhisperLinuxE2ETest] Segment $index: [${segment.startTimeMs}ms - ${segment.endTimeMs}ms] ${segment.text}")
        }

        // Test that we can get full text
        // Note: Full text is only available after transcription, before close()
        // So we need to get it before closing in a real scenario
        println("[WhisperLinuxE2ETest] Test completed successfully")
    }

    @Test
    fun `whisper version and system info`() {
        // This test doesn't require a model, just the native library
        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        Assume.assumeTrue("Native loading is disabled", disableNativeLoad != "true")

        try {
            val version = Whisper.getVersion()
            println("[WhisperLinuxE2ETest] Whisper version: $version")
            assertNotNull(version)
            assertTrue(version.isNotEmpty())

            val systemInfo = Whisper.getSystemInfo()
            println("[WhisperLinuxE2ETest] System info: $systemInfo")
            assertNotNull(systemInfo)
        } catch (e: UnsatisfiedLinkError) {
            Assume.assumeTrue("Native library not available", false)
        }
    }

    @Test
    fun `whisper language utilities`() {
        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        Assume.assumeTrue("Native loading is disabled", disableNativeLoad != "true")

        try {
            val maxLangId = Whisper.getMaxLanguageId()
            println("[WhisperLinuxE2ETest] Max language ID: $maxLangId")
            assertTrue("Expected at least some languages", maxLangId > 0)

            val englishId = Whisper.getLanguageId("en")
            println("[WhisperLinuxE2ETest] English language ID: $englishId")
            assertTrue("Expected valid English ID", englishId >= 0)

            val langStr = Whisper.getLanguageString(englishId)
            println("[WhisperLinuxE2ETest] Language string for ID $englishId: $langStr")
            assertEquals("en", langStr)
        } catch (e: UnsatisfiedLinkError) {
            Assume.assumeTrue("Native library not available", false)
        }
    }

    @Test
    fun `transcribe with real audio file`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        Assume.assumeTrue("No test model specified", !modelPath.isNullOrBlank())

        val audioPath = System.getenv("LLMEDGE_TEST_AUDIO_PATH") ?: System.getProperty("LLMEDGE_TEST_AUDIO_PATH")
        Assume.assumeTrue("No test audio file specified", !audioPath.isNullOrBlank())

        val audioFile = File(audioPath)
        Assume.assumeTrue("Audio file not found", audioFile.exists())

        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        Assume.assumeTrue("Native loading is disabled", disableNativeLoad != "true")

        println("[WhisperLinuxE2ETest] Testing with real audio file: $audioPath")

        // Load model
        val whisper = Whisper.load(modelPath, useGpu = false)

        // Read audio file - for this test, we expect a raw PCM float32 file
        // Real implementation should use proper audio decoding
        val audioBytes = audioFile.readBytes()
        val samples = FloatArray(audioBytes.size / 4)
        val buffer = java.nio.ByteBuffer.wrap(audioBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (i in samples.indices) {
            samples[i] = buffer.float
        }

        println("[WhisperLinuxE2ETest] Loaded ${samples.size} samples from audio file")

        // Transcribe with progress callback
        var progressReported = false
        whisper.setProgressCallback { progress ->
            println("[WhisperLinuxE2ETest] Progress: $progress%")
            progressReported = true
        }

        val segments = mutableListOf<Whisper.TranscriptionSegment>()
        whisper.setSegmentCallback { index, startTime, endTime, text ->
            println("[WhisperLinuxE2ETest] New segment: [$startTime - $endTime] $text")
            segments.add(Whisper.TranscriptionSegment(index, startTime, endTime, text))
        }

        val result = whisper.transcribe(samples)
        whisper.close()

        println("[WhisperLinuxE2ETest] Transcription complete: ${result.size} segments")
        result.forEach { segment ->
            println("[WhisperLinuxE2ETest] ${segment.toSrtEntry()}")
        }

        assertTrue("Expected at least one segment", result.isNotEmpty())
    }
}
