package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.runBlocking
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.io.FileOutputStream
import java.io.ByteArrayOutputStream

/**
 * E2E test that generates a video with a specific sampler and scheduler,
 * outputting results as GIF file for visual verification.
 * 
 * Run with environment variables:
 * LLMEDGE_BUILD_NATIVE_LIB_PATH=/path/to/libsdcpp.so
 * LLMEDGE_TEST_MODEL_PATH=/path/to/wan2.1_t2v_1.3B_fp16.safetensors
 * LLMEDGE_TEST_T5_PATH=/path/to/umt5-xxl-encoder-Q3_K_S.gguf
 * LLMEDGE_TEST_VAE_PATH=/path/to/wan_2.1_vae.safetensors
 * 
 * Optional - specify sampler/scheduler to test:
 * LLMEDGE_TEST_SAMPLER=EULER (default: DEFAULT)
 * LLMEDGE_TEST_SCHEDULER=KARRAS (default: DEFAULT)
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SamplerSchedulerGifTest {

    companion object {
        private const val MODEL_PATH_ENV = "LLMEDGE_TEST_MODEL_PATH"
        private const val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    }

    @Test
    fun `generate video with specified sampler and save as GIF`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        
        println("[SamplerGifTest] modelPath=$modelPath t5Path=$t5Path vaePath=$vaePath")
        Assume.assumeTrue("Model path not set", !modelPath.isNullOrBlank())
        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        val libPath = System.getenv(LIB_PATH_ENV)
            ?: System.getProperty(LIB_PATH_ENV)
            ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsdcpp.so"
        Assume.assumeTrue("Native library not found", File(libPath).exists())
        Assume.assumeTrue("Native loading disabled", System.getProperty("llmedge.disableNativeLoad") != "true")

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val outputDir = File("${System.getProperty("user.dir")}/sampler_scheduler_tests")
        outputDir.mkdirs()

        // Get sampler/scheduler from environment or use defaults
        val samplerName = System.getenv("LLMEDGE_TEST_SAMPLER") ?: "DEFAULT"
        val schedulerName = System.getenv("LLMEDGE_TEST_SCHEDULER") ?: "DEFAULT"
        
        val sampleMethod = try {
            SampleMethod.valueOf(samplerName)
        } catch (e: Exception) {
            SampleMethod.DEFAULT
        }
        
        val scheduler = try {
            Scheduler.valueOf(schedulerName)
        } catch (e: Exception) {
            Scheduler.DEFAULT
        }

        // Test parameters
        val width = 256
        val height = 256
        val videoFrames = 5  // Will produce exactly 5 frames
        val steps = System.getenv("LLMEDGE_TEST_STEPS")?.toIntOrNull() ?: 10
        val cfgScale = 7.0f
        val seed = 42L
        val prompt = "A cat walking in a garden"

        println("[SamplerGifTest] Testing sampler=$sampleMethod (id=${sampleMethod.id}), scheduler=$scheduler (id=${scheduler.id})")
        println("[SamplerGifTest] Parameters: ${width}x${height}, frames=$videoFrames, steps=$steps, cfg=$cfgScale, seed=$seed")

        // Load model
        println("[SamplerGifTest] Loading model...")
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
            sequentialLoad = true
        )

        try {
            val params = VideoGenerateParams(
                prompt = prompt,
                width = width,
                height = height,
                videoFrames = videoFrames,
                steps = steps,
                cfgScale = cfgScale,
                seed = seed,
                sampleMethod = sampleMethod,
                scheduler = scheduler
            )

            println("[SamplerGifTest] Starting video generation...")
            val bitmaps = sd.txt2vid(params)
            println("[SamplerGifTest] ✓ Generated ${bitmaps.size} frames")
            
            // Save as GIF
            val gifName = "test_${sampleMethod.name.lowercase()}_${scheduler.name.lowercase()}_${steps}steps.gif"
            val gifFile = File(outputDir, gifName)
            saveAsGif(bitmaps, gifFile, delayMs = 100)
            println("[SamplerGifTest] ✓ Saved: ${gifFile.absolutePath}")
            
        } finally {
            sd.close()
        }
    }

    private fun saveAsGif(bitmaps: List<Bitmap>, outputFile: File, delayMs: Int) {
        if (bitmaps.isEmpty()) return

        // Save individual frames as PNG first
        val framesDir = File(outputFile.parent, "${outputFile.nameWithoutExtension}_frames")
        framesDir.mkdirs()
        
        bitmaps.forEachIndexed { index, bitmap ->
            val frameFile = File(framesDir, "frame_${String.format("%03d", index)}.png")
            FileOutputStream(frameFile).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
            }
            println("[SamplerGifTest] Saved frame ${index + 1}/${bitmaps.size}: ${frameFile.name}")
        }
        
        // Use ffmpeg to create GIF if available
        try {
            val pb = ProcessBuilder(
                "ffmpeg", "-y",
                "-framerate", (1000 / delayMs).toString(),
                "-i", "${framesDir.absolutePath}/frame_%03d.png",
                "-vf", "palettegen=stats_mode=diff",
                "${framesDir.absolutePath}/palette.png"
            )
            pb.redirectErrorStream(true)
            val p1 = pb.start()
            p1.inputStream.bufferedReader().readText() // consume output
            p1.waitFor()

            val pb2 = ProcessBuilder(
                "ffmpeg", "-y",
                "-framerate", (1000 / delayMs).toString(),
                "-i", "${framesDir.absolutePath}/frame_%03d.png",
                "-i", "${framesDir.absolutePath}/palette.png",
                "-lavfi", "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-loop", "0",
                outputFile.absolutePath
            )
            pb2.redirectErrorStream(true)
            val p2 = pb2.start()
            p2.inputStream.bufferedReader().readText() // consume output
            p2.waitFor()
            
            println("[SamplerGifTest] ✓ GIF created with ffmpeg: ${outputFile.absolutePath}")
        } catch (e: Exception) {
            println("[SamplerGifTest] ffmpeg not available (${e.message}), frames saved as PNGs in: ${framesDir.absolutePath}")
        }
    }
}
