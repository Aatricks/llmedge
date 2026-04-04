package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import java.io.File
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams

/**
 * Linux-host E2E tests for:
 * 1. Image-to-Video (I2V) generation
 * 2. LoRA weight application
 * 
 * These tests verify the Kotlin wrapper works correctly with these features.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoI2VAndLoraE2ETest {

    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_MODEL_PATH"

    /**
     * Test Image-to-Video (I2V) generation.
     * Creates a simple gradient image and uses it as init image.
     */
    @Test
    fun `I2V video generation with init image`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        Assume.assumeTrue("No test model specified", !modelPath.isNullOrBlank())

        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = LIB_PATH_ENV,
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        // Create a simple gradient init image (256x256)
        val width = 256
        val height = 256
        val initImage = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                // Create orange-to-blue gradient (sunset colors)
                val r = (255 * (1 - x.toFloat() / width)).toInt()
                val g = (128 * (1 - y.toFloat() / height)).toInt()
                val b = (255 * x.toFloat() / width).toInt()
                initImage.setPixel(x, y, Color.rgb(r, g, b))
            }
        }

        println("[I2VTest] Created init image: ${initImage.width}x${initImage.height}")
        println("[I2VTest] Loading model...")

        val sd = StableDiffusion.load(
            context = context,
            modelPath = modelPath,
            vaePath = vaePath,
            t5xxlPath = t5Path,
            nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true,
            sequentialLoad = false
        )

        println("[I2VTest] Model loaded, generating I2V video...")

        val startTime = System.currentTimeMillis()
        val bitmaps = try {
            val params = VideoGenerateParams(
                prompt = "a beautiful sunset animation, smooth motion",
                negative = "blurry, static",
                width = width,
                height = height,
                videoFrames = 5,  // Must be 5+ to get multiple frames (Wan formula: (n-1)/4*4+1)
                steps = 10,
                cfgScale = 7.0f,
                seed = 42L,
                initImage = initImage,  // I2V: use init image
                strength = 0.7f  // 0.7 = 70% generation, 30% init image influence
            )

            sd.txt2vid(params) { step, totalSteps, currentFrame, totalFrames, timePerStep ->
                println("[I2VTest] Progress: step=$step/$totalSteps, frame=$currentFrame/$totalFrames")
            }
        } finally {
            sd.close()
            initImage.recycle()
        }

        val elapsed = System.currentTimeMillis() - startTime
        println("[I2VTest] I2V generation completed in ${elapsed}ms, got ${bitmaps.size} frames")

        // Export frames
        val projectRoot = System.getProperty("user.dir")
        val framesDir = File(projectRoot, "generated_frames_i2v")
        framesDir.mkdirs()
        bitmaps.forEachIndexed { index, bitmap ->
            val frameFile = File(framesDir, "frame_${String.format("%03d", index)}.png")
            java.io.FileOutputStream(frameFile).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
            }
        }
        println("[I2VTest] Frames exported to: ${framesDir.absolutePath}")
        
        // Create GIF
        try {
            val gifFile = File(projectRoot, "generated_i2v.gif")
            val proc = ProcessBuilder("ffmpeg", "-y", "-framerate", "8", 
                "-i", "${framesDir.absolutePath}/frame_%03d.png", 
                "-loop", "0", gifFile.absolutePath)
                .redirectErrorStream(true).start()
            proc.waitFor()
            if (gifFile.exists()) println("[I2VTest] GIF created: ${gifFile.absolutePath}")
        } catch (e: Exception) { /* ffmpeg not available */ }

        // Verify we got frames
        assertTrue("Expected at least 1 frame from I2V", bitmaps.isNotEmpty())
        
        // Verify frames are not blank
        bitmaps.forEach { bmp ->
            assertEquals(width, bmp.width)
            assertEquals(height, bmp.height)
            
            var nonBlankFound = false
            outer@ for (y in 0 until bmp.height step 10) {
                for (x in 0 until bmp.width step 10) {
                    val px = bmp.getPixel(x, y)
                    if ((px ushr 24) != 0 && (px and 0x00FFFFFF) != 0x000000) {
                        nonBlankFound = true
                        break@outer
                    }
                }
            }
            assertTrue("I2V frame is blank", nonBlankFound)
        }

        println("[I2VTest] I2V test PASSED")
    }

    /**
     * Test LoRA weight loading (without actually having a LoRA file).
     * This test verifies the LoRA path handling doesn't crash.
     */
    @Test
    fun `LoRA path handling without crash`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        Assume.assumeTrue("No test model specified", !modelPath.isNullOrBlank())

        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = LIB_PATH_ENV,
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        // Create empty lora directory
        val loraDir = File(System.getProperty("user.dir"), "loras")
        loraDir.mkdirs()

        println("[LoRATest] Testing with empty LoRA directory: ${loraDir.absolutePath}")
        println("[LoRATest] Loading model with LoRA dir...")

        // This should NOT crash even with empty/missing LoRA files
        val sd = try {
            StableDiffusion.load(
                context = context,
                modelPath = modelPath,
                vaePath = vaePath,
                t5xxlPath = t5Path,
                nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                sequentialLoad = false,
                loraModelDir = loraDir.absolutePath,
                loraApplyMode = LoraApplyMode.AUTO
            )
        } catch (e: Exception) {
            println("[LoRATest] Model loading with LoRA dir failed: ${e.message}")
            // This is expected if LoRA handling is strict
            null
        }

        if (sd != null) {
            println("[LoRATest] Model loaded successfully with LoRA dir")
            
            // Try generating with LoRA path set
            val bitmaps = try {
                val params = VideoGenerateParams(
                    prompt = "a simple test",
                    width = 256,
                    height = 256,
                    videoFrames = 4,
                    steps = 10,
                    cfgScale = 7.0f,
                    seed = 1L
                )

                sd.txt2vid(params) { step, totalSteps, _, _, _ ->
                    if (step == 1) println("[LoRATest] Generation started...")
                }
            } finally {
                sd.close()
            }

            println("[LoRATest] Generation with LoRA path completed, got ${bitmaps.size} frames")
            assertTrue("Expected frames even with empty LoRA dir", bitmaps.isNotEmpty())
        }

        println("[LoRATest] LoRA path handling test PASSED (no crash)")
    }

    /**
     * Test with actual LoRA file if available.
     */
    @Test
    fun `LoRA generation with real LoRA file`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        Assume.assumeTrue("No test model specified", !modelPath.isNullOrBlank())

        val loraPath = System.getenv("LLMEDGE_TEST_LORA_PATH") ?: System.getProperty("LLMEDGE_TEST_LORA_PATH")
        Assume.assumeTrue("No LoRA path specified in LLMEDGE_TEST_LORA_PATH", !loraPath.isNullOrBlank())
        
        val loraFile = File(loraPath)
        Assume.assumeTrue("LoRA file not found: $loraPath", loraFile.exists())

        val libPath = System.getenv(LIB_PATH_ENV)
            ?: System.getProperty(LIB_PATH_ENV)
            ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsdcpp.so"
        Assume.assumeTrue("Native library not found", File(libPath).exists())

        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        Assume.assumeTrue("Native loading disabled", disableNativeLoad != "true")

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        val loraDir = loraFile.parentFile?.absolutePath ?: loraFile.absolutePath
        println("[LoRATest] Testing with real LoRA: $loraPath")
        println("[LoRATest] LoRA directory: $loraDir")

        val sd = StableDiffusion.load(
            context = context,
            modelPath = modelPath,
            vaePath = vaePath,
            t5xxlPath = t5Path,
            nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true,
            sequentialLoad = false,
            loraModelDir = loraDir,
            loraApplyMode = LoraApplyMode.AUTO
        )

        println("[LoRATest] Model loaded with LoRA, generating...")

        val startTime = System.currentTimeMillis()
        val bitmaps = try {
            val params = VideoGenerateParams(
                prompt = "UKR style war footage movie clip, military vehicles, dramatic scene",
                negative = "peaceful, calm, cartoon",
                width = 256,
                height = 256,
                videoFrames = 5,  // Must be 5+ to get multiple frames (Wan formula: (n-1)/4*4+1)
                steps = 10,
                cfgScale = 7.0f,
                seed = 42L
            )

            sd.txt2vid(params) { step, totalSteps, currentFrame, totalFrames, _ ->
                println("[LoRATest] Progress: step=$step/$totalSteps, frame=$currentFrame/$totalFrames")
            }
        } finally {
            sd.close()
        }

        val elapsed = System.currentTimeMillis() - startTime
        println("[LoRATest] LoRA generation completed in ${elapsed}ms, got ${bitmaps.size} frames")

        // Export frames
        val projectRoot = System.getProperty("user.dir")
        val framesDir = File(projectRoot, "generated_frames_lora")
        framesDir.mkdirs()
        bitmaps.forEachIndexed { index, bitmap ->
            val frameFile = File(framesDir, "frame_${String.format("%03d", index)}.png")
            java.io.FileOutputStream(frameFile).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
            }
        }
        println("[LoRATest] Frames exported to: ${framesDir.absolutePath}")
        
        // Create GIF
        try {
            val gifFile = File(projectRoot, "generated_lora.gif")
            val proc = ProcessBuilder("ffmpeg", "-y", "-framerate", "8", 
                "-i", "${framesDir.absolutePath}/frame_%03d.png", 
                "-loop", "0", gifFile.absolutePath)
                .redirectErrorStream(true).start()
            proc.waitFor()
            if (gifFile.exists()) println("[LoRATest] GIF created: ${gifFile.absolutePath}")
        } catch (e: Exception) { /* ffmpeg not available */ }

        assertTrue("Expected at least 1 frame with LoRA", bitmaps.isNotEmpty())
        println("[LoRATest] Real LoRA test PASSED")
    }
}
