package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.StableDiffusion
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

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ImageGenerationLinuxE2ETest {
    private val libPathEnv = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    private val modelPathEnv = "LLMEDGE_TEST_MODEL_PATH"

    @Test
    fun `desktop end-to-end image generation`() = runBlocking {
        val modelPath = System.getenv(modelPathEnv) ?: System.getProperty(modelPathEnv)
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        val taesdPath =
            System.getenv("LLMEDGE_TEST_TAESD_PATH") ?: System.getProperty("LLMEDGE_TEST_TAESD_PATH")

        Assume.assumeTrue("Model path not set", !modelPath.isNullOrBlank())
        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE or TAESD path not set", !vaePath.isNullOrBlank() || !taesdPath.isNullOrBlank())

        val libPath =
            DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
                envName = libPathEnv,
                defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
            )

        val width = 128
        val height = 128
        val steps = 8
        val cfg = 6.0f
        val seed = 42L
        val prompt = "a simple test image with clear colors"

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val loadStart = System.currentTimeMillis()
        val sd =
            StableDiffusion.load(
                context = context,
                modelPath = modelPath,
                vaePath = vaePath,
                t5xxlPath = t5Path,
                taesdPath = taesdPath,
                nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                sequentialLoad = false,
            )
        val loadMs = System.currentTimeMillis() - loadStart

        val genStart = System.currentTimeMillis()
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
        val genMs = System.currentTimeMillis() - genStart

        assertEquals(width, bitmap.width)
        assertEquals(height, bitmap.height)
        assertEquals(Bitmap.Config.ARGB_8888, bitmap.config)

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val uniqueColors = pixels.toSet().size
        val nonBlank = pixels.any { (it ushr 24) != 0 && (it and 0x00FFFFFF) != 0x000000 }
        assertTrue("Image should not be blank", nonBlank)
        assertTrue("Expected >1 unique color, got $uniqueColors", uniqueColors > 1)

        val outputDir = File("build/outputs/images")
        outputDir.mkdirs()
        val outputFile = File(outputDir, "image_e2e_${width}x${height}_s${steps}_seed${seed}.png")
        FileOutputStream(outputFile).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }

        println(
            "[ImageGenerationLinuxE2ETest] loadMs=$loadMs genMs=$genMs uniqueColors=$uniqueColors output=${outputFile.absolutePath}",
        )
    }
}
