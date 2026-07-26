package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Desktop (host) E2E for the Stable Diffusion 3 Medium split-model path.
 *
 * Provide the component files via env / system properties:
 *   LLMEDGE_TEST_SD3_DIFFUSION_PATH
 *   LLMEDGE_TEST_SD3_CLIP_L_PATH
 *   LLMEDGE_TEST_SD3_CLIP_G_PATH
 *   LLMEDGE_TEST_SD3_VAE_PATH
 *   LLMEDGE_TEST_SD3_LORA_PATH (optional)
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class Sd3MediumLinuxE2ETest {
    private val libPathEnv = "LLMEDGE_BUILD_NATIVE_LIB_PATH"

    private fun env(name: String): String? = System.getenv(name) ?: System.getProperty(name)

    @Test
    fun `desktop end-to-end SD3 Medium split-model generation`() = runBlocking {
        val diffusionPath = env("LLMEDGE_TEST_SD3_DIFFUSION_PATH")
        val clipLPath = env("LLMEDGE_TEST_SD3_CLIP_L_PATH")
        val clipGPath = env("LLMEDGE_TEST_SD3_CLIP_G_PATH")
        val vaePath = env("LLMEDGE_TEST_SD3_VAE_PATH")
        val loraPath = env("LLMEDGE_TEST_SD3_LORA_PATH")

        Assume.assumeTrue("Diffusion model path not set", !diffusionPath.isNullOrBlank())
        Assume.assumeTrue("CLIP_L path not set", !clipLPath.isNullOrBlank())
        Assume.assumeTrue("CLIP_G path not set", !clipGPath.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = libPathEnv,
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )

        val width = 256
        val height = 256
        val loraFile = loraPath?.let(::File)
        if (loraFile != null) {
            Assume.assumeTrue("LoRA file not found", loraFile.isFile)
        }
        val steps = if (loraFile != null) 4 else 28
        val cfg = if (loraFile != null) 3.0f else 4.5f
        val seed = 42L
        val prompt =
            if (loraFile != null) {
                "a red fox in snow, detailed, 8k <lora:${loraFile.nameWithoutExtension}:0.125>"
            } else {
                "a red fox in snow, detailed, 8k"
            }

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val sd =
            StableDiffusion.load(
                context = context,
                diffusionModelPath = diffusionPath,
                vaePath = vaePath,
                nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                sequentialLoad = false,
                loraModelDir = loraFile?.parentFile?.absolutePath,
                loraApplyMode = LoraApplyMode.AUTO,
                componentPaths = StableDiffusionComponentPaths(
                    clipLPath = clipLPath,
                    clipGPath = clipGPath,
                ),
            )

        val bitmap =
            try {
                sd.txt2img(
                    GenerateParams(
                        prompt = prompt,
                        negative = "",
                        width = width,
                        height = height,
                        steps = steps,
                        cfgScale = cfg,
                        seed = seed,
                    ),
                )
            } finally {
                sd.close()
            }

        assertEquals(width, bitmap.width)
        assertEquals(height, bitmap.height)

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val uniqueColors = pixels.toSet().size
        val nonBlank = pixels.any { (it ushr 24) != 0 && (it and 0x00FFFFFF) != 0x000000 }
        assertTrue("Image should not be blank", nonBlank)
        assertTrue("Expected >1 unique color, got $uniqueColors", uniqueColors > 1)

        val outputDir = File("build/outputs/images")
        outputDir.mkdirs()
        val outputPrefix = if (loraFile != null) "hyper_sd3" else "sd3_medium"
        val outputFile = File(outputDir, "${outputPrefix}_e2e_${width}x${height}_s${steps}_seed${seed}.png")
        FileOutputStream(outputFile).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }

        println("[Sd3MediumLinuxE2ETest] uniqueColors=$uniqueColors output=${outputFile.absolutePath}")
    }
}
