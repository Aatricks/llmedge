package io.aatricks.llmedge

import android.content.Context
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Full stack test for Wan 2.1 T2V video generation WITH T5XXL encoder.
 *
 * This test verifies the complete text-to-video pipeline including:
 * - Wan 2.1 T2V 1.3B model (bf16)
 * - VAE decoder
 * - T5XXL text encoder (Q3_K_S)
 *
 * Run via adb: adb shell am instrument -w -e class io.aatricks.llmedge.WanVideoFullStackTest \
 * io.aatricks.llmedge.test/androidx.test.runner.AndroidJUnitRunner
 */
@LargeTest
@RunWith(AndroidJUnit4::class)
class WanVideoFullStackTest {

    @Test
    fun testFullWanVideoGenerationWithT5XXL() {
        runBlocking {
            assumeTrue("Requires arm64 device", Build.SUPPORTED_ABIS.any { it.contains("arm64") })
            assumeTrue("Native library not loaded", StableDiffusion.isNativeLibraryLoaded())

            val instrumentation = InstrumentationRegistry.getInstrumentation()
            val targetContext = instrumentation.targetContext

            android.util.Log.i(TAG, "========================================")
            android.util.Log.i(TAG, "Wan 2.1 T2V Full Stack Test WITH T5XXL")
            android.util.Log.i(TAG, "========================================")

            logMemoryInfo(targetContext, "INITIAL")

            // Download all required files
            android.util.Log.i(TAG, "Downloading model files...")
            val files = downloadModelFiles(targetContext)

            android.util.Log.i(TAG, "Model files ready:")
            android.util.Log.i(
                    TAG,
                    "  Model: ${files.model.absolutePath} (${files.model.length() / 1024 / 1024}MB)"
            )
            android.util.Log.i(
                    TAG,
                    "  VAE: ${files.vae.absolutePath} (${files.vae.length() / 1024 / 1024}MB)"
            )
            android.util.Log.i(
                    TAG,
                    "  T5XXL: ${files.t5xxl.absolutePath} (${files.t5xxl.length() / 1024 / 1024}MB)"
            )

            logMemoryInfo(targetContext, "AFTER downloads")

            android.util.Log.i(TAG, "Loading StableDiffusion with full stack...")
            val loadStartTime = System.currentTimeMillis()

            try {
                val engine =
                        StableDiffusion.load(
                                context = targetContext,
                                modelPath = files.model.absolutePath,
                                vaePath = files.vae.absolutePath,
                                t5xxlPath = files.t5xxl.absolutePath, // *** INCLUDE T5XXL ***
                                nThreads =
                                        io.aatricks.llmedge.CpuTopology.getOptimalThreadCount(
                                                        io.aatricks.llmedge.CpuTopology.TaskType
                                                                .DIFFUSION
                                                )
                                                .coerceAtMost(4),
                                offloadToCpu = true, // Use CPU to avoid OOM
                                keepClipOnCpu = true, // Keep T5XXL on CPU to save VRAM
                                keepVaeOnCpu = true, // Keep VAE on CPU to save VRAM
                        )

                val loadDuration = System.currentTimeMillis() - loadStartTime
                android.util.Log.i(TAG, "✓ Full stack loaded successfully in ${loadDuration}ms")
                logMemoryInfo(targetContext, "AFTER model loading")

                engine.use { sd ->
                    assertTrue("Model should be video-capable", sd.isVideoModel())
                    android.util.Log.i(TAG, "✓ Model is video-capable")

                    // Use minimal parameters to reduce memory pressure
                    val params =
                            VideoGenerateParams(
                                    prompt = "a cat walking",
                                    width = 256,
                                    height = 256,
                                    videoFrames = 4,
                                    steps = 10,
                                    cfgScale = 7.0f,
                                    seed = 42L,
                                    sampleMethod = SampleMethod.EULER_A
                            )

                    android.util.Log.i(TAG, "Generation parameters:")
                    android.util.Log.i(TAG, "  Prompt: ${params.prompt}")
                    android.util.Log.i(TAG, "  Resolution: ${params.width}x${params.height}")
                    android.util.Log.i(TAG, "  Frames: ${params.videoFrames}")
                    android.util.Log.i(TAG, "  Steps: ${params.steps}")
                    android.util.Log.i(TAG, "  CFG Scale: ${params.cfgScale}")

                    logMemoryInfo(targetContext, "BEFORE video generation")

                    android.util.Log.i(TAG, "Starting video generation with T5XXL...")
                    val genStartTime = System.currentTimeMillis()

                    val frames = sd.txt2vid(params)

                    val genDuration = System.currentTimeMillis() - genStartTime
                    android.util.Log.i(TAG, "✓ Video generation completed in ${genDuration}ms")
                    logMemoryInfo(targetContext, "AFTER video generation")

                    // Verify results
                    assertNotNull("Frames should not be null", frames)
                    assertEquals(
                            "Should generate ${params.videoFrames} frames",
                            params.videoFrames,
                            frames.size
                    )
                    android.util.Log.i(TAG, "✓ Generated ${frames.size} frames")

                    frames.forEachIndexed { index, bitmap ->
                        assertNotNull("Frame $index should not be null", bitmap)
                        assertEquals("Frame $index width", params.width, bitmap.width)
                        assertEquals("Frame $index height", params.height, bitmap.height)
                        android.util.Log.i(TAG, "✓ Frame $index: ${bitmap.width}x${bitmap.height}")
                    }

                    android.util.Log.i(TAG, "========================================")
                    android.util.Log.i(TAG, "✅ FULL STACK TEST PASSED!")
                    android.util.Log.i(TAG, "========================================")
                    android.util.Log.i(TAG, "Text-to-video generation with T5XXL works!")
                    android.util.Log.i(
                            TAG,
                            "Total time: ${(System.currentTimeMillis() - loadStartTime) / 1000}s"
                    )
                    android.util.Log.i(TAG, "========================================")

                    // Clean up frames
                    frames.forEach { it.recycle() }
                }
            } catch (e: Exception) {
                android.util.Log.e(TAG, "========================================")
                android.util.Log.e(TAG, "❌ TEST FAILED", e)
                android.util.Log.e(TAG, "========================================")
                logMemoryInfo(targetContext, "AFTER failure")
                throw e
            }
        }
    }

    private suspend fun downloadModelFiles(context: Context): ModelFiles =
            withContext(Dispatchers.IO) {
                android.util.Log.i(TAG, "Downloading main model (bf16)...")
                val modelResult =
                        HuggingFaceHub.ensureRepoFileOnDisk(
                                context = context,
                                modelId = MODEL_REPO,
                                revision = "main",
                                filename = MODEL_FILENAME,
                                allowedExtensions = listOf(".safetensors"),
                                preferSystemDownloader = true,
                        )
                android.util.Log.i(TAG, "✓ Model downloaded: ${modelResult.file.absolutePath}")

                android.util.Log.i(TAG, "Downloading VAE...")
                val vaeResult =
                        HuggingFaceHub.ensureRepoFileOnDisk(
                                context = context,
                                modelId = MODEL_REPO,
                                revision = "main",
                                filename = VAE_FILENAME,
                                allowedExtensions = listOf(".safetensors"),
                                preferSystemDownloader = true,
                        )
                android.util.Log.i(TAG, "✓ VAE downloaded: ${vaeResult.file.absolutePath}")

                android.util.Log.i(TAG, "Downloading T5XXL encoder (Q3_K_S)...")
                val t5xxlResult =
                        HuggingFaceHub.ensureRepoFileOnDisk(
                                context = context,
                                modelId = T5XXL_REPO,
                                revision = "main",
                                filename = T5XXL_FILENAME,
                                allowedExtensions = listOf(".gguf"),
                                preferSystemDownloader = true,
                        )
                android.util.Log.i(TAG, "✓ T5XXL downloaded: ${t5xxlResult.file.absolutePath}")

                ModelFiles(model = modelResult.file, vae = vaeResult.file, t5xxl = t5xxlResult.file)
            }

    private fun logMemoryInfo(context: Context, label: String) {
        val activityManager =
                context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memoryInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)

        val totalMB = memoryInfo.totalMem / 1024 / 1024
        val availMB = memoryInfo.availMem / 1024 / 1024
        val usedMB = totalMB - availMB
        val threshold = memoryInfo.threshold / 1024 / 1024
        val lowMemory = memoryInfo.lowMemory

        val runtime = Runtime.getRuntime()
        val javaMaxMB = runtime.maxMemory() / 1024 / 1024
        val javaUsedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024

        android.util.Log.i(TAG, "--- Memory Info: $label ---")
        android.util.Log.i(TAG, "System Memory:")
        android.util.Log.i(TAG, "  Total: ${totalMB}MB")
        android.util.Log.i(TAG, "  Available: ${availMB}MB")
        android.util.Log.i(TAG, "  Used: ${usedMB}MB (${usedMB * 100 / totalMB}%)")
        android.util.Log.i(TAG, "  Threshold: ${threshold}MB")
        android.util.Log.i(TAG, "  Low memory: $lowMemory")
        android.util.Log.i(TAG, "Java Heap:")
        android.util.Log.i(TAG, "  Max: ${javaMaxMB}MB")
        android.util.Log.i(TAG, "  Used: ${javaUsedMB}MB (${javaUsedMB * 100 / javaMaxMB}%)")
        android.util.Log.i(TAG, "---")
    }

    private companion object {
        const val TAG = "WanVideoFullStack"
        const val MODEL_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
        const val MODEL_FILENAME = "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"
        const val VAE_FILENAME = "split_files/vae/wan_2.1_vae.safetensors"
        const val T5XXL_REPO = "city96/umt5-xxl-encoder-gguf"
        const val T5XXL_FILENAME = "umt5-xxl-encoder-Q3_K_S.gguf"
    }

    private data class ModelFiles(
            val model: File,
            val vae: File,
            val t5xxl: File,
    )
}
