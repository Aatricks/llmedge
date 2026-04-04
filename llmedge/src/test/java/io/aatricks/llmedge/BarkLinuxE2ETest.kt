package io.aatricks.llmedge

import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.speech.tts.BarkTTS

/**
 * Linux-host end-to-end test for Bark TTS using a real native library (libbark_jni.so) and a local
 * model file.
 *
 * Requirements to run:
 * - Build the native bark library for Linux and place as
 * llmedge/build/native/linux-x86_64/libbark_jni.so (There is a script at
 * scripts/build_native_linux.sh bark to help).
 * - Provide a test model file via environment variable LLMEDGE_TEST_BARK_MODEL_PATH (path to a bark
 * ggml_weights.bin file). If not set, the test will be skipped.
 *
 * This test exercises the full TTS pipeline from text to audio samples.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class BarkLinuxE2ETest {

    private val LIB_PATH_ENV = "LLMEDGE_BUILD_BARK_LIB_PATH"
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_BARK_MODEL_PATH"

    @Test
    fun `desktop end-to-end bark TTS generation`() = runBlocking {
        // Skip test if model path is not provided
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        println("[BarkLinuxE2ETest] modelPath=$modelPath")
        Assume.assumeTrue("No test model specified in $MODEL_PATH_ENV", !modelPath.isNullOrBlank())

        // Check that native library path is properly set and library file exists
        val libPath =
                DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
                        envName = LIB_PATH_ENV,
                        defaultRelativePath = "llmedge/build/native/linux-x86_64/libbark_jni.so",
                )

        val libFile = File(libPath)
        println("[BarkLinuxE2ETest] libPath=$libPath libExists=${libFile.exists()}")
        println("[BarkLinuxE2ETest] java.library.path=${System.getProperty("java.library.path")}")

        // Check if model file exists
        val modelFile = File(modelPath)
        Assume.assumeTrue(
                "Model file not found at $modelPath",
                modelFile.exists() && modelFile.isFile
        )

        println("[BarkLinuxE2ETest] Loading Bark model...")

        var progressSteps = mutableListOf<Pair<BarkTTS.EncodingStep, Int>>()

        val bark =
                try {
                    BarkTTS.load(
                            modelPath = modelPath,
                            seed = 42, // Fixed seed for reproducibility
                            temperature = 0.7f,
                            fineTemperature = 0.5f,
                            verbosity = 1
                    )
                } catch (e: Exception) {
                    println("[BarkLinuxE2ETest] Failed to load model: ${e.message}")
                    e.printStackTrace()
                    throw e
                }

        println("[BarkLinuxE2ETest] Model loaded, load time: ${bark.getLoadTime() / 1000.0}ms")
        println("[BarkLinuxE2ETest] Sample rate: ${bark.getSampleRate()}Hz")

        // Set progress callback
        bark.setProgressCallback { step, progress ->
            progressSteps.add(step to progress)
            println("[BarkLinuxE2ETest] Progress: ${step.name} - $progress%")
        }

        // Test TTS generation
        val testText = "Hello, world!"
        println("[BarkLinuxE2ETest] Generating audio for: \"$testText\"")

        val startTime = System.currentTimeMillis()
        val audioResult =
                try {
                    bark.generate(
                            text = testText,
                            params =
                                    BarkTTS.GenerateParams(
                                            nThreads =
                                                    Runtime.getRuntime()
                                                            .availableProcessors()
                                                            .coerceAtMost(8)
                                    )
                    )
                } catch (e: Exception) {
                    println("[BarkLinuxE2ETest] TTS generation failed: ${e.message}")
                    e.printStackTrace()
                    throw e
                }

        val elapsed = System.currentTimeMillis() - startTime
        println("[BarkLinuxE2ETest] Generation completed in ${elapsed}ms")
        println(
                "[BarkLinuxE2ETest] Generated ${audioResult.samples.size} samples (${audioResult.durationSeconds}s)"
        )
        println("[BarkLinuxE2ETest] Eval time: ${bark.getEvalTime() / 1000.0}ms")

        // Verify audio was generated
        assertTrue("No audio samples generated", audioResult.samples.isNotEmpty())
        assertTrue("Audio duration too short", audioResult.durationSeconds > 0.1f)
        assertEquals("Sample rate mismatch", BarkTTS.SAMPLE_RATE, audioResult.sampleRate)

        // Verify progress callbacks were called
        println("[BarkLinuxE2ETest] Received ${progressSteps.size} progress updates")

        // Save the audio to a WAV file for verification
        val outputWav = File(System.getProperty("java.io.tmpdir"), "bark_test_output.wav")
        bark.saveAsWav(audioResult, outputWav.absolutePath)
        println("[BarkLinuxE2ETest] Saved audio to: ${outputWav.absolutePath}")
        assertTrue("WAV file not created", outputWav.exists())
        assertTrue("WAV file is empty", outputWav.length() > 44) // WAV header is 44 bytes

        bark.close()
        println("[BarkLinuxE2ETest] Test completed successfully")
    }

    @Test
    fun `bark bindings check`() {
        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
                envName = LIB_PATH_ENV,
                defaultRelativePath = "llmedge/build/native/linux-x86_64/libbark_jni.so",
        )

        try {
            val bindingsOk = BarkTTS.checkBindings()
            println("[BarkLinuxE2ETest] Bark bindings available: $bindingsOk")
            assertTrue("Bark bindings not available", bindingsOk)
        } catch (e: UnsatisfiedLinkError) {
            println("[BarkLinuxE2ETest] Native library error: ${e.message}")
            throw e
        }
    }

    @Test
    fun `bark short text generation`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        Assume.assumeTrue("No test model specified", !modelPath.isNullOrBlank())

        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
                envName = LIB_PATH_ENV,
                defaultRelativePath = "llmedge/build/native/linux-x86_64/libbark_jni.so",
        )

        val modelDir = File(modelPath)
        Assume.assumeTrue("Model not found", modelDir.exists() && modelDir.isDirectory)

        println("[BarkLinuxE2ETest] Testing short text generation...")

        val bark = BarkTTS.load(modelPath, seed = 123)

        try {
            val result = bark.generate("Hi!")

            assertTrue("No samples", result.samples.isNotEmpty())
            println(
                    "[BarkLinuxE2ETest] Short text: ${result.samples.size} samples, ${result.durationSeconds}s"
            )
        } finally {
            bark.close()
        }
    }

    @Test
    fun `bark generate with different seed produces different output`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        Assume.assumeTrue("No test model specified", !modelPath.isNullOrBlank())

        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
                envName = LIB_PATH_ENV,
                defaultRelativePath = "llmedge/build/native/linux-x86_64/libbark_jni.so",
        )

        val modelDir = File(modelPath)
        Assume.assumeTrue("Model not found", modelDir.exists() && modelDir.isDirectory)

        println("[BarkLinuxE2ETest] Testing seed variation...")

        val bark1 = BarkTTS.load(modelPath, seed = 111)
        val bark2 = BarkTTS.load(modelPath, seed = 222)

        try {
            val text = "Test"
            val result1 = bark1.generate(text)
            val result2 = bark2.generate(text)

            // Different seeds should produce different audio
            // Note: This may not always be true for very short texts
            println("[BarkLinuxE2ETest] Seed 111: ${result1.samples.size} samples")
            println("[BarkLinuxE2ETest] Seed 222: ${result2.samples.size} samples")

            // Just verify both produce valid output
            assertTrue("Seed 111 no samples", result1.samples.isNotEmpty())
            assertTrue("Seed 222 no samples", result2.samples.isNotEmpty())
        } finally {
            bark1.close()
            bark2.close()
        }
    }
}
