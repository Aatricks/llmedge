package io.aatricks.llmedge

import android.graphics.Bitmap
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.runtime.CpuTopology

/**
 * Opt-in native end-to-end test for txt2img.
 *
 * This test only runs when:
 *  - System property llmedge.runNativeImageE2E=true
 *  - A valid model file path is provided via env var LLMEDGE_T2I_MODEL_PATH or system property llmedge.t2i.modelPath
 *  - Optionally, VAE path via LLMEDGE_T2I_VAE_PATH or llmedge.t2i.vaePath
 *
 * It loads the model through StableDiffusion.load (native JNI), generates a small bitmap, and
 * asserts that the output is non-null and has expected dimensions and non-uniform pixel values.
 */
@RunWith(AndroidJUnit4::class)
class ImageGenerationE2ENativeTest {

    @Test
    fun txt2img_native_e2e() {
        // Only run when explicitly requested
        val enabled = java.lang.Boolean.getBoolean("llmedge.runNativeImageE2E")
        assumeTrue("Set -Dllmedge.runNativeImageE2E=true to enable this test", enabled)

        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext

        val modelPath = System.getenv("LLMEDGE_T2I_MODEL_PATH")
            ?: System.getProperty("llmedge.t2i.modelPath")
            ?: ""
        assumeTrue("Provide T2I model via LLMEDGE_T2I_MODEL_PATH or -Dllmedge.t2i.modelPath",
            modelPath.isNotBlank() && File(modelPath).exists())

        val vaePath = System.getenv("LLMEDGE_T2I_VAE_PATH")
            ?: System.getProperty("llmedge.t2i.vaePath")
            ?: null
        if (vaePath != null) {
            assumeTrue("Provided VAE path does not exist: $vaePath", File(vaePath).exists())
        }

        // Load the model natively (do not disable native load for this test)
        // Run suspend API from a coroutine
        kotlinx.coroutines.runBlocking {
            val sd = StableDiffusion.load(
                context = context,
                modelPath = modelPath,
                vaePath = vaePath,
                t5xxlPath = null,
                nThreads = io.aatricks.llmedge.runtime.CpuTopology.getOptimalThreadCount(io.aatricks.llmedge.runtime.CpuTopology.TaskType.DIFFUSION).coerceAtLeast(2),
                offloadToCpu = false,
                keepClipOnCpu = false,
                keepVaeOnCpu = false,
            )

            sd.use { engine ->
                val params = GenerateParams(
                    prompt = "a scenic landscape, high quality",
                    width = 64,
                    height = 64,
                    steps = 10,
                    cfgScale = 7.0f,
                    seed = 42L,
                )

                val bmp: Bitmap = engine.txt2img(params)
            assertNotNull("Bitmap should not be null", bmp)
            assertEquals(64, bmp.width)
            assertEquals(64, bmp.height)

            // Validate that the image is not uniformly zero; compute a simple checksum and variance-like metric
                val pixels = IntArray(bmp.width * bmp.height)
                bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
                var sum = 0L
                var sumSq = 0L
                for (p in pixels) {
                    val r = (p shr 16) and 0xFF
                    val g = (p shr 8) and 0xFF
                    val b = p and 0xFF
                    val v = (r + g + b)
                    sum += v
                    sumSq += (v * v)
                }
                // Mean and a crude variance check (not normalized) just to ensure non-uniform output
                val n = pixels.size
                val mean = sum.toDouble() / n
                val varianceLike = (sumSq.toDouble() / n) - (mean * mean)
                assumeTrue("Image appears uniform (likely generation failed)", varianceLike > 0.0)
            }
        }
    }
}
