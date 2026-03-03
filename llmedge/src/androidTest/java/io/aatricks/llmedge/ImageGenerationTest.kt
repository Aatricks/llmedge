package io.aatricks.llmedge

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ImageGenerationTest {

    @Before
    fun setUp() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        // Provide a stub bridge that returns a deterministic RGB buffer for validation
        StableDiffusion.overrideNativeBridgeForTests { _ ->
            object : StableDiffusion.NativeBridge {
                override fun txt2img(
                        handle: Long,
                        prompt: String,
                        negative: String,
                        width: Int,
                        height: Int,
                        steps: Int,
                        cfg: Float,
                        seed: Long,
                        vaeTiling: Boolean,
                        easyCacheEnabled: Boolean,
                        easyCacheReuseThreshold: Float,
                        easyCacheStartPercent: Float,
                        easyCacheEndPercent: Float,
                ): ByteArray {
                    // Solid gradient pattern independent of prompt
                    val bytes = ByteArray(width * height * 3)
                    var idx = 0
                    val wDen = (width - 1).coerceAtLeast(1)
                    val hDen = (height - 1).coerceAtLeast(1)
                    for (y in 0 until height) {
                        for (x in 0 until width) {
                            val r = (x * 255 / wDen).toByte()
                            val g = (y * 255 / hDen).toByte()
                            val b = 0.toByte()
                            bytes[idx++] = r
                            bytes[idx++] = g
                            bytes[idx++] = b
                        }
                    }
                    return bytes
                }

                override fun txt2vid(
                        handle: Long,
                        prompt: String,
                        negative: String,
                        width: Int,
                        height: Int,
                        videoFrames: Int,
                        steps: Int,
                        cfg: Float,
                        seed: Long,
                        sampleMethod: StableDiffusion.SampleMethod,
                        scheduler: StableDiffusion.Scheduler,
                        strength: Float,
                        initImage: ByteArray?,
                        initWidth: Int,
                        initHeight: Int,
                        easyCacheEnabled: Boolean,
                        easyCacheReuseThreshold: Float,
                        easyCacheStartPercent: Float,
                        easyCacheEndPercent: Float,
                ): Array<ByteArray>? {
                    // Not used in this test
                    return null
                }

                override fun setProgressCallback(
                        handle: Long,
                        callback: StableDiffusion.VideoProgressCallback?
                ) {}
                override fun cancelGeneration(handle: Long) {}
                override fun precomputeCondition(
                        handle: Long,
                        prompt: String,
                        negative: String,
                        width: Int,
                        height: Int,
                        clipSkip: Int,
                ): StableDiffusion.PrecomputedCondition? = null
            }
        }
    }

    @After
    fun tearDown() {
        StableDiffusion.resetNativeBridgeForTests()
    }

    @Test
    fun txt2img_convertsRgbToBitmapCorrectly() = runBlocking {
        // Construct an instance without invoking native init via reflection
        val ctor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        ctor.isAccessible = true
        val sd = ctor.newInstance(1L)

        val params =
                StableDiffusion.GenerateParams(
                        prompt = "test",
                        width = 8,
                        height = 4,
                        steps = 5,
                        cfgScale = 7.0f,
                        seed = 42L,
                )

        val bmp: Bitmap = sd.txt2img(params)
        assertNotNull(bmp)
        assertEquals(8, bmp.width)
        assertEquals(4, bmp.height)

        // Top-left pixel should be (0,0,0)
        val p00 = bmp.getPixel(0, 0)
        assertEquals(0, (p00 shr 16) and 0xFF)
        assertEquals(0, (p00 shr 8) and 0xFF)
        assertEquals(0, (p00) and 0xFF)

        // Right-most pixel on first row should have R=255, G=0
        val p70 = bmp.getPixel(7, 0)
        assertEquals(255, (p70 shr 16) and 0xFF)
        assertEquals(0, (p70 shr 8) and 0xFF)
        assertEquals(0, (p70) and 0xFF)

        // Bottom-left pixel should have R=0, G=255
        val p03 = bmp.getPixel(0, 3)
        assertEquals(0, (p03 shr 16) and 0xFF)
        assertEquals(255, (p03 shr 8) and 0xFF)
        assertEquals(0, (p03) and 0xFF)

        sd.close()
    }
}
