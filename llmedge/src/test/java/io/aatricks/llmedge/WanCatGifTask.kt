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
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class WanCatGifTask {

    @Test
    fun generateCatGifs() = runBlocking {
        val projectRoot = System.getProperty("user.dir").let {
            if (it.endsWith("/llmedge")) File(it).parentFile else File(it)
        }
        val modelDir = File(projectRoot, "models")
        val modelPath = File(modelDir, "wan2.1_t2v_1.3B_fp16.safetensors").absolutePath
        val vaePath = File(modelDir, "wan_2.1_vae.safetensors").absolutePath
        val t5Path = File(modelDir, "umt5-xxl-encoder-Q3_K_S.gguf").absolutePath
        val taehvPath = File(modelDir, "taew2_1.safetensors").absolutePath

        Assume.assumeTrue("WAN model file is missing", File(modelPath).exists())
        Assume.assumeTrue("WAN VAE file is missing", File(vaePath).exists())
        Assume.assumeTrue("WAN T5 file is missing", File(t5Path).exists())
        Assume.assumeTrue("WAN TAEHV file is missing", File(taehvPath).exists())

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val prompt = "a cute cat, high quality, 4k"
        
        // 1. Path using full VAE
        println("--- PATH 1: Using Full VAE ---")
        generateGif(context, "cat_vae.gif", modelPath, vaePath, t5Path, null, prompt)

        // 2. Path using TAEHV (Tiny AutoEncoder)
        println("\n--- PATH 2: Using TAEHV ---")
        generateGif(context, "cat_taehv.gif", modelPath, null, t5Path, taehvPath, prompt)
    }

    private suspend fun generateGif(
        context: Context,
        outputName: String,
        modelPath: String,
        vaePath: String?,
        t5Path: String?,
        taesdPath: String?,
        prompt: String
    ) {
        println("[Step 1] Loading StableDiffusion model components...")
        // Kotlin: StableDiffusion.load calls nativeCreate
        // JNI: Java_io_aatricks_llmedge_StableDiffusion_nativeCreate
        // C++: new_sd_ctx initializes DiT (Wan), T5, and VAE/TAE
        val sd = StableDiffusion.load(
            context = context,
            modelPath = modelPath,
            vaePath = vaePath,
            t5xxlPath = t5Path,
            taesdPath = taesdPath,
            nThreads = 4,
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true
        )

        println("[Step 2] Generating video frames...")
        // Kotlin: StableDiffusion.txt2vid calls nativeTxt2Vid
        // JNI: Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Vid
        // C++: generate_video performs diffusion and decoding
        val params = VideoGenerateParams(
            prompt = prompt,
            width = 256,
            height = 256,
            videoFrames = 5,
            steps = 1,
            cfgScale = 7.0f,
            seed = 42L
        )

        val bitmaps = sd.txt2vid(params) { step, total, curFrame, totalFrames, _ ->
            println("  Progress: Step $step/$total, Frame $curFrame/$totalFrames")
        }

        println("[Step 3] Post-processing: Saving GIF...")
        // Bitmaps are created from RGB bytes returned by JNI
        val outputFile = File(outputName)
        FileOutputStream(outputFile).use {
            io.aatricks.llmedge.vision.ImageUtils.createAnimatedGif(
                frames = bitmaps,
                delayMs = 125,
                output = it
            )
        }
        println("[Done] GIF saved to ${outputFile.absolutePath}")

        sd.close()
    }
}
