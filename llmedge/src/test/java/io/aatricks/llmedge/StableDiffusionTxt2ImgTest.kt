package io.aatricks.llmedge

import android.graphics.Color
import kotlinx.coroutines.test.runTest
import org.junit.runner.RunWith
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StableDiffusionTxt2ImgTest {
    @Before
    fun setUp() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
    }

    @After
    fun tearDown() {
        StableDiffusion.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `txt2img converts RGB bytes to Bitmap correctly`() = runTest {
        val width = 2
        val height = 2
        val rgb = byteArrayOf(
            0x10.toByte(), 0x20.toByte(), 0x30.toByte(),  // (0,0)
            0x40.toByte(), 0x50.toByte(), 0x60.toByte(),  // (1,0)
            0x70.toByte(), 0x80.toByte(), 0x90.toByte(),  // (0,1)
            0xAA.toByte(), 0xBB.toByte(), 0xCC.toByte(),  // (1,1)
        )

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
                ): ByteArray? = rgb

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
                    vaceStrength: Float,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
                ): Array<ByteArray>? = null

                override fun setProgressCallback(handle: Long, callback: StableDiffusion.VideoProgressCallback?) {}
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

        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)

        val bmp = sd.txt2img(StableDiffusion.GenerateParams(prompt = "hi", width = width, height = height, steps = 1))
        assertEquals(width, bmp.width)
        assertEquals(height, bmp.height)
        assertEquals(Color.rgb(0x10, 0x20, 0x30), bmp.getPixel(0, 0))
        assertEquals(Color.rgb(0x40, 0x50, 0x60), bmp.getPixel(1, 0))
        assertEquals(Color.rgb(0x70, 0x80, 0x90), bmp.getPixel(0, 1))
        assertEquals(Color.rgb(0xAA, 0xBB, 0xCC), bmp.getPixel(1, 1))
    }
}
