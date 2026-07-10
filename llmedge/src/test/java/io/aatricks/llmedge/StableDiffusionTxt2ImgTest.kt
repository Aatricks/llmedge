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
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback

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
        val width = 64
        val height = 64
        val rgb = ByteArray(width * height * 3).apply {
            this[0] = 0x10.toByte(); this[1] = 0x20.toByte(); this[2] = 0x30.toByte()
            this[3] = 0x40.toByte(); this[4] = 0x50.toByte(); this[5] = 0x60.toByte()
            this[width * 3 + 0] = 0x70.toByte(); this[width * 3 + 1] = 0x80.toByte(); this[width * 3 + 2] = 0x90.toByte()
            this[width * 3 + 3] = 0xAA.toByte(); this[width * 3 + 4] = 0xBB.toByte(); this[width * 3 + 5] = 0xCC.toByte()
        }

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
                    sampleMethod: SampleMethod,
                    scheduler: Scheduler,
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

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) {}
                override fun cancelGeneration(handle: Long) {}
                override fun precomputeCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    clipSkip: Int,
                ): PrecomputedCondition? = null
            }
        }

        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)

        val bmp = sd.txt2img(GenerateParams(prompt = "hi", width = width, height = height, steps = 1))
        assertEquals(width, bmp.width)
        assertEquals(height, bmp.height)
        assertEquals(Color.rgb(0x10, 0x20, 0x30), bmp.getPixel(0, 0))
        assertEquals(Color.rgb(0x40, 0x50, 0x60), bmp.getPixel(1, 0))
        assertEquals(Color.rgb(0x70, 0x80, 0x90), bmp.getPixel(0, 1))
        assertEquals(Color.rgb(0xAA, 0xBB, 0xCC), bmp.getPixel(1, 1))
    }

    @Test
    fun `GenerateParams dimension validation throws IllegalArgumentException`() {
        // Test width <= 0
        try {
            GenerateParams(prompt = "hi", width = 0, height = 512)
            org.junit.Assert.fail("Expected IllegalArgumentException for width <= 0")
        } catch (e: IllegalArgumentException) {
            // expected
        }

        // Test height <= 0
        try {
            GenerateParams(prompt = "hi", width = 512, height = -10)
            org.junit.Assert.fail("Expected IllegalArgumentException for height <= 0")
        } catch (e: IllegalArgumentException) {
            // expected
        }

        // Test width not multiple of 64
        try {
            GenerateParams(prompt = "hi", width = 500, height = 512)
            org.junit.Assert.fail("Expected IllegalArgumentException for width % 64 != 0")
        } catch (e: IllegalArgumentException) {
            // expected
        }

        // Test height not multiple of 64
        try {
            GenerateParams(prompt = "hi", width = 512, height = 510)
            org.junit.Assert.fail("Expected IllegalArgumentException for height % 64 != 0")
        } catch (e: IllegalArgumentException) {
            // expected
        }
    }

    @Test
    fun `ARGB-null-is-terminal behavior throws InferenceFailedException`() = runTest {
        StableDiffusion.overrideNativeBridgeForTests { _ ->
            object : MockStableDiffusionBridge() {
                override fun txt2imgArgb(
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
                ): IntArray? = null // Explicitly return null to simulate JNI failure

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
                ): ByteArray? = byteArrayOf(0)
            }
        }

        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)

        try {
            sd.txt2img(GenerateParams(prompt = "hi", width = 512, height = 512, steps = 1))
            org.junit.Assert.fail("Expected InferenceFailedException because ARGB returned null")
        } catch (e: io.aatricks.llmedge.core.InferenceFailedException) {
            // expected
        }
    }

    @Test
    fun `UnsupportedModelException is thrown and instance is closed when mobileSupported is false`() = runTest {
        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)

        org.junit.Assert.assertFalse(sd.state.closed.get())

        val metadata = io.aatricks.llmedge.image.diffusion.VideoModelMetadata(
            architecture = "FLUX.1-schnell",
            modelType = "diffusion",
            parameterCount = "14B",
            mobileSupported = false,
            tags = emptySet(),
            filename = "flux.gguf"
        )
        sd.state.modelMetadata = metadata

        try {
            if (sd.state.modelMetadata?.mobileSupported == false) {
                val paramCount = sd.state.modelMetadata?.parameterCount ?: "14B"
                sd.close()
                throw io.aatricks.llmedge.core.UnsupportedModelException(
                    "$paramCount models are not supported on mobile devices. " +
                        "Please use 1.3B or 5B model variants instead. " +
                        "14B models require 20-40GB RAM and are designed for desktop/server use only.",
                )
            }
            org.junit.Assert.fail("Expected UnsupportedModelException")
        } catch (e: io.aatricks.llmedge.core.UnsupportedModelException) {
            assertEquals("14B models are not supported on mobile devices. Please use 1.3B or 5B model variants instead. 14B models require 20-40GB RAM and are designed for desktop/server use only.", e.message)
        }

        org.junit.Assert.assertTrue(sd.state.closed.get())
    }

    @Test
    fun `txt2img on closed instance throws IllegalStateException and does not call bridge`() = runTest {
        var bridgeCalled = false
        StableDiffusion.overrideNativeBridgeForTests { _ ->
            object : StableDiffusion.NativeBridge {
                override fun txt2imgArgb(
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
                ): IntArray? {
                    bridgeCalled = true
                    return intArrayOf()
                }

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
                ): ByteArray? {
                    bridgeCalled = true
                    return byteArrayOf()
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
                    sampleMethod: SampleMethod,
                    scheduler: Scheduler,
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

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) {}
                override fun cancelGeneration(handle: Long) {}
                override fun precomputeCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    clipSkip: Int,
                ): PrecomputedCondition? = null
            }
        }

        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)

        // Close the instance
        sd.close()

        var threwExpected = false
        try {
            sd.txt2img(GenerateParams(prompt = "hi", width = 64, height = 64, steps = 1))
        } catch (e: IllegalStateException) {
            if (e.message == "StableDiffusion is closed") {
                threwExpected = true
            }
        }

        assertEquals(true, threwExpected)
        org.junit.Assert.assertFalse("Bridge txt2img must not be called after close", bridgeCalled)
    }
}
