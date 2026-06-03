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

/**
 * Desktop (host) E2E for the FLUX.2 Klein split-model path: a standalone diffusion transformer
 * (routed to diffusion_model_path), a Qwen3 LLM text encoder (llm_path) and the FLUX.2 VAE. This
 * exercises the same JNI routing the Android runtime uses, but on the host so it sidesteps the
 * Apple-Silicon emulator PAC quirk.
 *
 * Provide the three component files via env / system properties:
 *   LLMEDGE_TEST_DIFFUSION_PATH  flux-2-klein-4b-*.gguf
 *   LLMEDGE_TEST_LLM_PATH        qwen_3_4b*.gguf (or .safetensors)
 *   LLMEDGE_TEST_VAE_PATH        flux2-vae.safetensors
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class Flux2KleinLinuxE2ETest {
    private val libPathEnv = "LLMEDGE_BUILD_NATIVE_LIB_PATH"

    private fun env(name: String): String? = System.getenv(name) ?: System.getProperty(name)

    @Test
    fun `desktop DiT-only context loads without a text encoder`() = runBlocking {
        val diffusionPath = env("LLMEDGE_TEST_DIFFUSION_PATH")
        val vaePath = env("LLMEDGE_TEST_VAE_PATH")
        Assume.assumeTrue("Diffusion model path not set", !diffusionPath.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())
        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = libPathEnv,
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )
        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        // No llmPath: the FLUX.2 conditioner must be skipped (sequential / precomputed mode),
        // otherwise loading the DiT-only context fails on missing text_encoders.llm tensors.
        val sd =
            StableDiffusion.load(
                context = context,
                diffusionModelPath = diffusionPath,
                vaePath = vaePath,
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
            )
        sd.close()
        // Reaching here (no ModelLoadException) means the DiT-only context constructed.
    }

    @Test
    fun `desktop sequential precompute then DiT-only generation`() = runBlocking {
        val diffusionPath = env("LLMEDGE_TEST_DIFFUSION_PATH")
        val llmPath = env("LLMEDGE_TEST_LLM_PATH")
        val vaePath = env("LLMEDGE_TEST_VAE_PATH")
        Assume.assumeTrue("Diffusion path not set", !diffusionPath.isNullOrBlank())
        Assume.assumeTrue("LLM path not set", !llmPath.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())
        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = libPathEnv,
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )
        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val width = 256
        val height = 256
        val prompt = "a red fox in snow, detailed, 8k"

        // Phase 1: ENCODER-ONLY context (no DiT loaded) -> precompute the conditioning, then free it.
        // This is the memory win: peak here is the encoder alone, not encoder+DiT.
        val encoderCtx =
            StableDiffusion.load(
                context = context,
                llmPath = llmPath,
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
            )
        val cond = encoderCtx.precomputeCondition(prompt, "", width, height, -1)
        encoderCtx.close()
        assertTrue("precomputeCondition returned null", cond != null)

        // Phase 2: DiT-only context (NO text encoder loaded) -> generate from the precomputed cond.
        val ditCtx =
            StableDiffusion.load(
                context = context,
                diffusionModelPath = diffusionPath,
                vaePath = vaePath,
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
            )
        val rgb =
            try {
                ditCtx.txt2ImgWithPrecomputedCondition(
                    prompt = prompt,
                    negative = "",
                    width = width,
                    height = height,
                    steps = 4,
                    cfg = 1.0f,
                    seed = 42L,
                    cond = cond,
                    uncond = null,
                )
            } finally {
                ditCtx.close()
            }
        assertTrue("generation returned null", rgb != null)

        // rgb is width*height*3 (RGB). Build a bitmap + save for visual inspection.
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val px = IntArray(width * height)
        for (i in 0 until width * height) {
            val r = rgb!![i * 3].toInt() and 0xFF
            val g = rgb[i * 3 + 1].toInt() and 0xFF
            val b = rgb[i * 3 + 2].toInt() and 0xFF
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bmp.setPixels(px, 0, width, 0, 0, width, height)
        val uniqueColors = px.toSet().size
        assertTrue("Image should not be blank (uniqueColors=$uniqueColors)", uniqueColors > 1)
        val outDir = File("build/outputs/images")
        outDir.mkdirs()
        val out = File(outDir, "flux2_sequential_${width}x${height}_seed42.png")
        FileOutputStream(out).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        println("[Flux2KleinLinuxE2ETest sequential] uniqueColors=$uniqueColors output=${out.absolutePath}")
    }

    @Test
    fun `desktop end-to-end FLUX2 Klein split-model generation`() = runBlocking {
        val diffusionPath = env("LLMEDGE_TEST_DIFFUSION_PATH")
        val llmPath = env("LLMEDGE_TEST_LLM_PATH")
        val vaePath = env("LLMEDGE_TEST_VAE_PATH")

        Assume.assumeTrue("Diffusion model path not set", !diffusionPath.isNullOrBlank())
        Assume.assumeTrue("LLM encoder path not set", !llmPath.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = libPathEnv,
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )

        val width = 256
        val height = 256
        val steps = 4
        val cfg = 1.0f
        val seed = 42L
        val prompt = "a red fox in snow, detailed, 8k"

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val sd =
            StableDiffusion.load(
                context = context,
                diffusionModelPath = diffusionPath,
                vaePath = vaePath,
                llmPath = llmPath,
                nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                sequentialLoad = false,
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
        val outputFile = File(outputDir, "flux2_klein_e2e_${width}x${height}_s${steps}_seed${seed}.png")
        FileOutputStream(outputFile).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }

        println("[Flux2KleinLinuxE2ETest] uniqueColors=$uniqueColors output=${outputFile.absolutePath}")
    }
}
