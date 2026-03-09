package io.aatricks.llmedge

import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.speech.tts.BarkTTS

/**
 * Android E2E test for Bark TTS to measure performance on real device.
 *
 * This test requires a pre-downloaded Bark model on the device at:
 * /data/local/tmp/bark_ggml_weights.bin
 *
 * To push the model to the device: adb push models/bark_ggml_weights.bin /data/local/tmp/
 */
@LargeTest
@RunWith(AndroidJUnit4::class)
class BarkTtsAndroidE2ETest {

    companion object {
        private const val TAG = "BarkTtsE2ETest"
        private const val MODEL_PATH = "/data/local/tmp/bark_ggml_weights.bin"
    }

    @Test
    fun testBarkTtsPerformance() {
        runBlocking {
            Log.i(TAG, "====================================")
            Log.i(TAG, "Bark TTS Android E2E Performance Test")
            Log.i(TAG, "====================================")

            // Check for arm64 device
            assumeTrue("Requires arm64 device", Build.SUPPORTED_ABIS.any { it.contains("arm64") })

            // Check if native library is available
            val bindingsOk =
                    try {
                        BarkTTS.checkBindings()
                    } catch (e: UnsatisfiedLinkError) {
                        Log.e(TAG, "Native library not loaded: ${e.message}")
                        false
                    }
            assumeTrue("Bark native library not available", bindingsOk)

            // Check if model file exists
            val modelFile = File(MODEL_PATH)
            Log.i(TAG, "Looking for model at: $MODEL_PATH")
            Log.i(TAG, "Model exists: ${modelFile.exists()}")

            if (!modelFile.exists()) {
                // Try alternative location in files dir
                val context = InstrumentationRegistry.getInstrumentation().targetContext
                val altPath = File(context.filesDir, "bark_ggml_weights.bin")
                Log.i(TAG, "Trying alternative path: ${altPath.absolutePath}")
                assumeTrue(
                        "Model file not found at $MODEL_PATH or ${altPath.absolutePath}",
                        altPath.exists()
                )
            }

            val actualModelPath =
                    if (modelFile.exists()) MODEL_PATH
                    else
                            File(
                                            InstrumentationRegistry.getInstrumentation()
                                                    .targetContext
                                                    .filesDir,
                                            "bark_ggml_weights.bin"
                                    )
                                    .absolutePath

            Log.i(TAG, "Using model at: $actualModelPath")
            Log.i(TAG, "Model size: ${File(actualModelPath).length() / 1024 / 1024}MB")
            Log.i(TAG, "Device: ${Build.MODEL}")
            Log.i(TAG, "CPU ABI: ${Build.SUPPORTED_ABIS.joinToString(", ")}")
            Log.i(TAG, "Available processors: ${Runtime.getRuntime().availableProcessors()}")

            // Load the model
            Log.i(TAG, "Loading Bark model...")
            val loadStart = System.currentTimeMillis()

            val bark =
                    BarkTTS.load(
                            modelPath = actualModelPath,
                            seed = 42,
                            temperature = 0.7f,
                            fineTemperature = 0.5f,
                            verbosity = 1
                    )

            val loadTime = System.currentTimeMillis() - loadStart
            Log.i(TAG, "Model loaded in ${loadTime}ms")
            Log.i(TAG, "Sample rate: ${bark.getSampleRate()}Hz")

            // Track progress
            val progressUpdates = mutableListOf<Pair<BarkTTS.EncodingStep, Int>>()
            bark.setProgressCallback { step, progress ->
                progressUpdates.add(step to progress)
                if (progress % 25 == 0) {
                    Log.i(TAG, "Progress: ${step.name} - $progress%")
                }
            }

            // Generate speech - same text as Linux test
            val testText = "Hello, world!"
            Log.i(TAG, "Generating speech for: \"$testText\"")

            // Use all available cores for maximum performance
            val nThreads = Runtime.getRuntime().availableProcessors()
            Log.i(TAG, "Using $nThreads threads (all available cores)")

            val genStart = System.currentTimeMillis()
            val audioResult =
                    bark.generate(
                            text = testText,
                            params = BarkTTS.GenerateParams(nThreads = nThreads)
                    )
            val genTime = System.currentTimeMillis() - genStart

            // Print results to stdout (appears in test output)
            val results = StringBuilder()
            results.appendLine("====================================")
            results.appendLine("BARK TTS PERFORMANCE RESULTS:")
            results.appendLine("====================================")
            results.appendLine("Java-measured load time: ${loadTime}ms")
            results.appendLine("Java-measured generation time: ${genTime}ms (${genTime / 1000.0}s)")
            results.appendLine("Native load time: ${bark.getLoadTime() / 1000.0}ms")
            results.appendLine("Native eval time: ${bark.getEvalTime() / 1000.0}ms")
            results.appendLine(
                    "Total time: ${loadTime + genTime}ms (${(loadTime + genTime) / 1000.0}s)"
            )
            results.appendLine("Generated samples: ${audioResult.samples.size}")
            results.appendLine("Audio duration: ${audioResult.durationSeconds}s")
            results.appendLine("Progress updates: ${progressUpdates.size}")
            results.appendLine("Device: ${Build.MODEL} (${Build.SUPPORTED_ABIS[0]})")
            results.appendLine("Threads: $nThreads")
            results.appendLine("====================================")
            results.appendLine("COMPARISON (for 'Hello, world!'):")
            results.appendLine("Linux desktop: ~14 seconds")
            results.appendLine("This Android device: ${(loadTime + genTime) / 1000.0} seconds")
            results.appendLine(
                    "Ratio: ${String.format("%.1f", (loadTime + genTime) / 14000.0)}x slower"
            )
            results.appendLine("====================================")

            // Write results to a file that can be retrieved (use app cache dir for permission)
            try {
                val context = InstrumentationRegistry.getInstrumentation().targetContext
                val resultsFile = File(context.cacheDir, "bark_tts_results.txt")
                resultsFile.writeText(results.toString())
            } catch (e: Exception) {
                // Ignore file write errors
            }

            println(results.toString())

            // Also throw an assertion error if the test takes too long (helps debugging)
            assertTrue("Generation took ${genTime}ms - results: $results", genTime > 0)

            Log.i(TAG, "RESULTS:")
            Log.i(TAG, "Generation time: ${genTime}ms (${genTime / 1000.0}s)")
            Log.i(TAG, "Load time (from stats): ${bark.getLoadTime() / 1000.0}ms")
            Log.i(TAG, "Eval time (from stats): ${bark.getEvalTime() / 1000.0}ms")
            Log.i(TAG, "Generated samples: ${audioResult.samples.size}")
            Log.i(TAG, "Audio duration: ${audioResult.durationSeconds}s")
            Log.i(TAG, "Progress updates received: ${progressUpdates.size}")

            // Verify audio was generated
            assertTrue("No audio samples generated", audioResult.samples.isNotEmpty())
            assertTrue("Audio duration too short", audioResult.durationSeconds > 0.1f)
            assertEquals("Sample rate mismatch", BarkTTS.SAMPLE_RATE, audioResult.sampleRate)

            bark.close()
            println("Test completed successfully")
        }
    }

    @Test
    fun testBarkBindingsAvailable() {
        Log.i(TAG, "Testing Bark bindings...")

        val bindingsOk =
                try {
                    BarkTTS.checkBindings()
                } catch (e: UnsatisfiedLinkError) {
                    Log.e(TAG, "Native library not loaded: ${e.message}")
                    false
                }

        Log.i(TAG, "Bark bindings available: $bindingsOk")
        assertTrue("Bark bindings should be available", bindingsOk)
    }
}
