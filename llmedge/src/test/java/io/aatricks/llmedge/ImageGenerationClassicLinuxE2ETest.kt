package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Host E2E for classic single-file checkpoints (sd-turbo). Unlike
 * [ImageGenerationLinuxE2ETest] this needs no external T5/VAE files, so it is
 * cheap enough to run in CI, and it generates TWICE on the same instance:
 * warm sd_ctx reuse was untested for every runtime until it crashed on device
 * (see 863491a), so the second generation is the actual regression assertion.
 *
 * Requires:
 *   LLMEDGE_TEST_SD_MODEL_PATH  — classic checkpoint gguf (e.g. sd_turbo-f16-q8_0.gguf)
 *   LLMEDGE_BUILD_NATIVE_LIB_PATH — host libsdcpp (defaults to the linux build dir)
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ImageGenerationClassicLinuxE2ETest {

    @Test
    fun `desktop classic checkpoint generates twice on a warm context`() = runBlocking {
        val modelPath =
            System.getenv("LLMEDGE_TEST_SD_MODEL_PATH")
                ?: System.getProperty("LLMEDGE_TEST_SD_MODEL_PATH")
        Assume.assumeTrue("SD model path not set", !modelPath.isNullOrBlank())

        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = "LLMEDGE_BUILD_NATIVE_LIB_PATH",
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )

        val width = 128
        val height = 128
        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context

        val loadStart = System.currentTimeMillis()
        val sd =
            StableDiffusion.load(
                context = context,
                modelPath = modelPath,
                nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
            )
        val loadMs = System.currentTimeMillis() - loadStart

        try {
            repeat(2) { run ->
                val genStart = System.currentTimeMillis()
                val bitmap =
                    sd.txt2img(
                        GenerateParams(
                            prompt = "a red apple on a wooden table",
                            width = width,
                            height = height,
                            steps = 2,
                            cfgScale = 1.0f,
                            seed = 42L + run,
                        ),
                    )
                val genMs = System.currentTimeMillis() - genStart

                assertEquals(width, bitmap.width)
                assertEquals(height, bitmap.height)
                val pixels = IntArray(width * height)
                bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
                val uniqueColors = pixels.toSet().size
                assertTrue(
                    "Run $run: image should not be blank",
                    pixels.any { (it ushr 24) != 0 && (it and 0x00FFFFFF) != 0 },
                )
                assertTrue("Run $run: expected >1 unique color, got $uniqueColors", uniqueColors > 1)
                println(
                    "[ImageGenerationClassicLinuxE2ETest] run=$run loadMs=$loadMs genMs=$genMs uniqueColors=$uniqueColors",
                )
            }
        } finally {
            sd.close()
        }
    }
}
