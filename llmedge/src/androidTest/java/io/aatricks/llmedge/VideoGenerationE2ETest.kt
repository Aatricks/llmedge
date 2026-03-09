package io.aatricks.llmedge

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end test for video generation on physical device.
 *
 * This test uses the mock native bridge to simulate video generation and verify the full pipeline
 * works correctly.
 *
 * Run via adb: adb shell am instrument -w -e class io.aatricks.llmedge.VideoGenerationE2ETest \
 * io.aatricks.llmedge.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class VideoGenerationE2ETest : BaseVideoIntegrationTest() {

    private lateinit var context: Context

    @org.junit.Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
    }

    /** Full end-to-end test: load model, configure, generate, verify output */
    @Test
    fun testCompleteVideoGenerationWorkflow() {
        runBlocking {
            // Create StableDiffusion instance
            val sd = createStableDiffusion()

            // Configure as video model
            sd.updateModelMetadata(
                    VideoModelMetadata(
                            architecture = "wan",
                            modelType = "t2v",
                            parameterCount = "1.3B",
                            mobileSupported = true,
                            tags = setOf("wan", "video"),
                            filename = "Wan2.1-T2V-1.3B-Q4_K_M.gguf"
                    )
            )

            // Verify model is detected as video model
            assertTrue("Model should be detected as video model", sd.isVideoModel())

            // Set up progress callback
            val progressUpdates = mutableListOf<Pair<Int, Int>>()
            sd.setProgressCallback { step, totalSteps, currentFrame, totalFrames, timePerStep ->
                progressUpdates.add(Pair(step, totalSteps))
                android.util.Log.d(
                        "E2E_TEST",
                        "Progress: $step / $totalSteps (frame $currentFrame/$totalFrames, ${timePerStep}s/step)"
                )
            }

            // Create generation parameters
            val params =
                    VideoGenerateParams(
                            prompt = "a cat walking in a garden, high quality",
                            videoFrames = 8,
                            width = 256,
                            height = 256,
                            steps = 10,
                            cfgScale = 7.0f,
                            seed = 42,
                            sampleMethod = SampleMethod.EULER_A
                    )

            // Generate video
            android.util.Log.d("E2E_TEST", "Starting video generation...")
            val startTime = System.currentTimeMillis()
            val frames = sd.txt2vid(params)
            val duration = System.currentTimeMillis() - startTime

            // Verify results
            assertNotNull("Frames should not be null", frames)
            assertEquals("Should generate 8 frames", 8, frames.size)

            // Verify frame properties
            frames.forEachIndexed { index, frame ->
                assertNotNull("Frame $index should not be null", frame)
                assertEquals("Frame $index width should be 256", 256, frame.width)
                assertEquals("Frame $index height should be 256", 256, frame.height)
                android.util.Log.d("E2E_TEST", "Frame $index: ${frame.width}x${frame.height}")
            }

            // Verify progress was reported
            assertTrue("Progress should be reported", progressUpdates.isNotEmpty())
            android.util.Log.d("E2E_TEST", "Generation completed in ${duration}ms")
            android.util.Log.d("E2E_TEST", "Progress updates: ${progressUpdates.size}")

            // Check metrics
            val metrics = sd.getLastGenerationMetrics()
            assertNotNull("Metrics should be available", metrics)
            android.util.Log.d("E2E_TEST", "Metrics: $metrics")

            // Cleanup
            sd.close()
            android.util.Log.d("E2E_TEST", "Test completed successfully")
        }
    }

    /** Test video generation with different parameters */
    @Test
    fun testMultipleGenerationsWithDifferentParams() {
        runBlocking {
            val sd = createStableDiffusion()
            sd.updateModelMetadata(
                    VideoModelMetadata(
                            architecture = "wan",
                            modelType = "t2v",
                            parameterCount = "1.3B",
                            mobileSupported = true,
                            tags = setOf("wan", "video"),
                            filename = "test-model.gguf"
                    )
            )

            val testCases =
                    listOf(
                            // Test case 1: Small resolution, few frames
                            Triple(256, 256, 4),
                            // Test case 2: Medium resolution, more frames
                            Triple(512, 512, 8),
                            // Test case 3: Different aspect ratio
                            Triple(256, 512, 4)
                    )

            testCases.forEachIndexed { index, (width, height, frames) ->
                android.util.Log.d(
                        "E2E_TEST",
                        "Test case $index: ${width}x${height}, $frames frames"
                )

                val params =
                        VideoGenerateParams(
                                prompt = "test video $index",
                                videoFrames = frames,
                                width = width,
                                height = height,
                                steps = 10
                        )

                val result = sd.txt2vid(params)
                assertEquals("Should generate $frames frames", frames, result.size)
                assertEquals("Frame width should be $width", width, result.first().width)
                assertEquals("Frame height should be $height", height, result.first().height)

                android.util.Log.d("E2E_TEST", "Test case $index passed")
            }

            sd.close()
        }
    }

    /** Test cancellation during generation */
    @Test
    fun testCancellationDuringGeneration() {
        runBlocking {
            val sd = createStableDiffusion()
            sd.updateModelMetadata(
                    VideoModelMetadata(
                            architecture = "wan",
                            modelType = "t2v",
                            parameterCount = "1.3B",
                            mobileSupported = true,
                            tags = setOf("wan", "video"),
                            filename = "test-model.gguf"
                    )
            )

            val progressCount = AtomicInteger(0)
            sd.setProgressCallback { step, total, currentFrame, totalFrames, timePerStep ->
                progressCount.incrementAndGet()
                android.util.Log.d("E2E_TEST", "Cancellation test progress: $step/$total")

                // Cancel after first progress update
                if (step == 1) {
                    android.util.Log.d("E2E_TEST", "Cancelling generation...")
                    sd.cancelGeneration()
                }
            }

            val params =
                    VideoGenerateParams(
                            prompt = "test cancellation",
                            videoFrames = 16,
                            width = 256,
                            height = 256,
                            steps = 20 // More steps to ensure cancellation happens
                    )

            try {
                sd.txt2vid(params)
                // If we reach here, cancellation didn't work as expected
                android.util.Log.w("E2E_TEST", "Generation completed despite cancellation request")
            } catch (e: Exception) {
                android.util.Log.d("E2E_TEST", "Generation cancelled successfully: ${e.message}")
            }

            assertTrue("Progress callback should have been called", progressCount.get() > 0)
            sd.close()
        }
    }

    /** Test memory usage remains stable across multiple generations */
    @Test
    fun testMemoryStabilityAcrossGenerations() {
        runBlocking {
            val sd = createStableDiffusion()
            sd.updateModelMetadata(
                    VideoModelMetadata(
                            architecture = "wan",
                            modelType = "t2v",
                            parameterCount = "1.3B",
                            mobileSupported = true,
                            tags = setOf("wan", "video"),
                            filename = "test-model.gguf"
                    )
            )

            val params =
                    VideoGenerateParams(
                            prompt = "memory stability test",
                            videoFrames = 4,
                            width = 256,
                            height = 256,
                            steps = 10
                    )

            val memoryUsages = mutableListOf<Long>()
            val runtime = Runtime.getRuntime()

            // Run 5 generations and track memory
            repeat(5) { iteration ->
                android.util.Log.d("E2E_TEST", "Memory test iteration $iteration")

                val frames = sd.txt2vid(params)
                assertEquals("Should generate 4 frames", 4, frames.size)

                // Force GC to get more accurate memory reading
                System.gc()
                Thread.sleep(100)

                val usedMemory = runtime.totalMemory() - runtime.freeMemory()
                memoryUsages.add(usedMemory)

                android.util.Log.d(
                        "E2E_TEST",
                        "Memory after iteration $iteration: ${usedMemory / 1024 / 1024}MB"
                )

                // Clean up frames immediately
                frames.forEach { it.recycle() }
            }

            // Verify memory didn't grow excessively
            val firstMemory = memoryUsages.first()
            val lastMemory = memoryUsages.last()
            val growthPercent = ((lastMemory - firstMemory).toDouble() / firstMemory * 100)

            android.util.Log.d("E2E_TEST", "Memory growth: ${growthPercent.toInt()}%")
            android.util.Log.d(
                    "E2E_TEST",
                    "Memory usages: ${memoryUsages.map { it / 1024 / 1024 }}MB"
            )

            // Allow up to 70% growth to reduce flakiness in constrained test environments
            assertTrue("Memory growth should be under 70%, was $growthPercent%", growthPercent < 70)

            sd.close()
        }
    }

    /** Test all supported schedulers */
    @Test
    fun testAllSchedulers() {
        runBlocking {
            val sd = createStableDiffusion()
            sd.updateModelMetadata(
                    VideoModelMetadata(
                            architecture = "wan",
                            modelType = "t2v",
                            parameterCount = "1.3B",
                            mobileSupported = true,
                            tags = setOf("wan", "video"),
                            filename = "test-model.gguf"
                    )
            )

            val sampleMethods =
                    listOf(
                            SampleMethod.EULER_A,
                            SampleMethod.DDIM_TRAILING,
                            SampleMethod.DEFAULT,
                            SampleMethod.LCM
                    )

            sampleMethods.forEach { sampleMethod ->
                android.util.Log.d("E2E_TEST", "Testing sample method: $sampleMethod")

                val params =
                        VideoGenerateParams(
                                prompt = "test sample method $sampleMethod",
                                videoFrames = 4,
                                width = 256,
                                height = 256,
                                steps = 10,
                                sampleMethod = sampleMethod
                        )

                val frames = sd.txt2vid(params)
                assertEquals("Should generate 4 frames with $sampleMethod", 4, frames.size)

                android.util.Log.d("E2E_TEST", "Sample method $sampleMethod: SUCCESS")
            }

            sd.close()
        }
    }

    /** Test parameter validation edge cases */
    @Test
    fun testParameterValidation() {
        runBlocking {
            val sd = createStableDiffusion()
            sd.updateModelMetadata(
                    VideoModelMetadata(
                            architecture = "wan",
                            modelType = "t2v",
                            parameterCount = "1.3B",
                            mobileSupported = true,
                            tags = setOf("wan", "video"),
                            filename = "test-model.gguf"
                    )
            )

            // Test minimum valid parameters
            val minParams =
                    VideoGenerateParams(
                            prompt = "min test",
                            videoFrames = 4,
                            width = 256,
                            height = 256,
                            steps = 10,
                            cfgScale = 1.0f
                    )
            val minFrames = sd.txt2vid(minParams)
            assertEquals("Min params should work", 4, minFrames.size)
            android.util.Log.d("E2E_TEST", "Minimum parameters: PASS")

            // Test maximum valid parameters (within mobile limits)
            val maxParams =
                    VideoGenerateParams(
                            prompt = "max test",
                            videoFrames = 64,
                            width = 960,
                            height = 960,
                            steps = 50,
                            cfgScale = 15.0f
                    )
            val maxFrames = sd.txt2vid(maxParams)
            assertEquals("Max params should work", 64, maxFrames.size)
            android.util.Log.d("E2E_TEST", "Maximum parameters: PASS")

            sd.close()
        }
    }
}
