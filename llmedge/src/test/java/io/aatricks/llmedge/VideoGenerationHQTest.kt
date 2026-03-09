package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import java.io.File
import java.io.FileOutputStream
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoGenerationHQTest {

    @Test
    fun testTaehvHQ() = runBlocking {
        println("--- STARTING HIGH QUALITY TAEHV TEST ---")
        
        val modelPath = System.getProperty("LLMEDGE_TEST_MODEL_PATH") ?: System.getenv("LLMEDGE_TEST_MODEL_PATH")
        val t5Path = System.getProperty("LLMEDGE_TEST_T5_PATH") ?: System.getenv("LLMEDGE_TEST_T5_PATH")
        val taesdPath = System.getProperty("LLMEDGE_TEST_TAESD_PATH") ?: System.getenv("LLMEDGE_TEST_TAESD_PATH")
        
        if (modelPath == null || t5Path == null || taesdPath == null) {
            println("Skipping test: Missing paths")
            println("  LLMEDGE_TEST_MODEL_PATH=${modelPath ?: "not set"}")
            println("  LLMEDGE_TEST_T5_PATH=${t5Path ?: "not set"}")
            println("  LLMEDGE_TEST_TAESD_PATH=${taesdPath ?: "not set"}")
            return@runBlocking
        }

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context

        // Determine llmedge project root (we're in llmedge/src/test/...)
        val projectRoot = File(System.getProperty("user.dir") ?: ".")
        println("Project root: ${projectRoot.absolutePath}")
        
        // Create frames output directory at project root
        val framesDir = File(projectRoot, "hq_frames")
        if (framesDir.exists()) {
            framesDir.deleteRecursively()
        }
        framesDir.mkdirs()
        println("Frames output directory: ${framesDir.absolutePath}")

        try {
            val sd = StableDiffusion.load(
                context = context,
                modelPath = modelPath,
                vaePath = null,  // Don't load VAE, use TAEHV only
                t5xxlPath = t5Path,
                taesdPath = taesdPath,
                nThreads = 8,
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                vaeDecodeOnly = true,
                sequentialLoad = false
            )
            
            val params = VideoGenerateParams(
                prompt = "A high quality close up of a magical glowing butterfly in a forest, highly detailed, cinematic",
                width = 256,
                height = 256,
                videoFrames = 9,
                steps = 30, // 30 steps for real quality
                cfgScale = 5.0f,
                seed = 12345
            )
            
            println("Generating HQ video (30 steps)... this will take a minute.")
            val frames = sd.txt2vid(params) { step, total, fCurrent, fTotal, tps ->
                if (step % 5 == 0) println("Progress: step $step/$total")
            }
            
            if (frames.isNotEmpty()) {
                println("Generated ${frames.size} frames")
                
                // Export individual frames as PNG for inspection
                frames.forEachIndexed { index, bitmap ->
                    val frameFile = File(framesDir, "frame_%04d.png".format(index))
                    FileOutputStream(frameFile).use { fos ->
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
                    }
                    println("Saved frame $index to ${frameFile.absolutePath}")
                    
                    // Log some frame info for debugging
                    val brightness = computeAverageBrightness(bitmap)
                    println("  Frame $index: ${bitmap.width}x${bitmap.height}, avg brightness: $brightness")
                }
                
                // Export GIF to project root
                val rootGif = File(projectRoot, "generated_video_hq.gif")
                FileOutputStream(rootGif).use { fos ->
                    io.aatricks.llmedge.vision.ImageUtils.createAnimatedGif(frames, 125, fos, 0)
                }
                println("HQ GIF saved to ${rootGif.absolutePath}")
                
                // Also try using ffmpeg if available for comparison (higher quality GIF)
                tryFfmpegGif(framesDir, File(projectRoot, "generated_video_hq_ffmpeg.gif"))
            }
            
            sd.close()
        } catch (e: Exception) {
            e.printStackTrace()
            throw e
        }
    }
    
    private fun computeAverageBrightness(bitmap: Bitmap): Double {
        var total = 0L
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            total += (r + g + b) / 3
        }
        return total.toDouble() / pixels.size
    }
    
    private fun tryFfmpegGif(framesDir: File, outputGif: File) {
        try {
            val fps = 8
            val ffmpegCmd = arrayOf(
                "ffmpeg",
                "-y",
                "-framerate", fps.toString(),
                "-i", "${framesDir.absolutePath}/frame_%04d.png",
                "-filter_complex",
                "[0:v]fps=$fps,split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
                "-loop", "0",
                outputGif.absolutePath
            )
            
            println("Attempting ffmpeg GIF export: ${ffmpegCmd.joinToString(" ")}")
            val process = ProcessBuilder(*ffmpegCmd).redirectErrorStream(true).start()
            val ffmpegOutput = process.inputStream.bufferedReader().readText()
            val exitCode = process.waitFor()
            
            if (exitCode == 0 && outputGif.exists()) {
                println("ffmpeg GIF saved to ${outputGif.absolutePath}")
            } else {
                println("ffmpeg failed with exit code $exitCode: $ffmpegOutput")
            }
        } catch (t: Throwable) {
            println("ffmpeg not available: ${t.message}")
        }
    }
}
