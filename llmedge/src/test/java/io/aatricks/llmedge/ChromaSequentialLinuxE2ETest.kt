package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
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

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ChromaSequentialLinuxE2ETest {
    private val libPathEnv = "LLMEDGE_BUILD_NATIVE_LIB_PATH"

    private fun env(name: String): String? = System.getenv(name) ?: System.getProperty(name)

    @Test
    fun `desktop chroma sequential precompute then DiT-only generation`() = runBlocking {
        val ditPath = env("LLMEDGE_TEST_CHROMA_DIT_PATH")
        val t5Path = env("LLMEDGE_TEST_CHROMA_T5_PATH")
        val vaePath = env("LLMEDGE_TEST_CHROMA_VAE_PATH")

        Assume.assumeTrue("Chroma DiT path not set", !ditPath.isNullOrBlank())
        Assume.assumeTrue("Chroma T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("Chroma VAE path not set", !vaePath.isNullOrBlank())

        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = libPathEnv,
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val width = 128
        val height = 128
        val prompt = "a beautiful cherry blossom branch, digital art"
        val negativePrompt = "blurry, low quality"

        val encoderCtx = StableDiffusion.load(
            context = context,
            t5xxlPath = t5Path,
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true,
            allowVulkan = false,
            componentPaths = StableDiffusionComponentPaths(chromaT5ConditionerOnly = true),
        )

        val cond = encoderCtx.precomputeCondition(prompt, "", width, height, -1)
        val uncond = encoderCtx.precomputeCondition(negativePrompt, "", width, height, -1)
        encoderCtx.close()

        assertTrue("positive conditioning returned null", cond != null)
        assertTrue("negative conditioning returned null", uncond != null)

        val ditCtx = StableDiffusion.load(
            context = context,
            diffusionModelPath = ditPath,
            vaePath = vaePath,
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true,
        )

        val rgb = try {
            ditCtx.txt2ImgWithPrecomputedCondition(
                prompt = prompt,
                negative = negativePrompt,
                width = width,
                height = height,
                steps = 1,
                cfg = 4.0f,
                seed = 42L,
                cond = cond,
                uncond = uncond,
            )
        } finally {
            ditCtx.close()
        }

        assertTrue("generation returned null", rgb != null)
        assertEquals("output dimensions mismatch", width * height * 3, rgb!!.size)

        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val px = IntArray(width * height)
        for (i in 0 until width * height) {
            val r = rgb[i * 3].toInt() and 0xFF
            val g = rgb[i * 3 + 1].toInt() and 0xFF
            val b = rgb[i * 3 + 2].toInt() and 0xFF
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bmp.setPixels(px, 0, width, 0, 0, width, height)

        val uniqueColors = px.toSet().size
        assertTrue("Image should not be blank (uniqueColors=$uniqueColors)", uniqueColors > 1)

        val outDir = File("build/outputs/images")
        outDir.mkdirs()
        val out = File(outDir, "chroma_sequential_${width}x${height}_seed42.png")
        FileOutputStream(out).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        println("[ChromaSequentialLinuxE2ETest sequential] uniqueColors=$uniqueColors output=${out.absolutePath}")
    }
}
