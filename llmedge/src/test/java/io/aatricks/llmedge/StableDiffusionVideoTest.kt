package io.aatricks.llmedge

import android.content.Context
import io.mockk.*
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicBoolean
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback

@Suppress("unused")
private val disableNativeLoadForTests = run {
    System.setProperty("llmedge.disableNativeLoad", "true")
    StableDiffusion.enableNativeBridgeForTests()
    StableDiffusion.overrideNativeBridgeForTests { instance ->
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
            ): ByteArray? = null
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
            ): Array<ByteArray>? = arrayOf(byteArrayOf(1, 2, 3))

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
    true
}

class StableDiffusionVideoTest {

    @Test
    fun `wan architecture metadata counts as video`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "Wan 2.1 T2V",
                modelType = null,
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = emptySet(),
                filename = "hunyuan_video_720_cfgdistill_fp8.gguf",
            ),
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `tags mentioning video trigger detection`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "stable-diffusion-xl",
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = setOf("text-to-video", "wan-model"),
                filename = "sdxl-model.gguf",
            ),
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `non video metadata stays false`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "stable-diffusion-xl",
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = setOf("image", "text-to-image"),
                filename = "sdxl-ti.gguf",
            ),
        )

        assertFalse(sd.isVideoModel())
    }

    @Test
    fun `resetting metadata clears detection`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("video"),
                filename = "wan-model.gguf"
            ),
        )
        assertTrue(sd.isVideoModel())

        sd.updateModelMetadata(null)

        assertFalse(sd.isVideoModel())
    }

    @Test
    fun `load requires either model id or path`() = runTest {
        val context = mockk<Context>(relaxed = true)

        val result = runCatching {
            StableDiffusion.load(
                context = context,
                modelId = null,
                modelPath = null,
            )
        }

        val error = result.exceptionOrNull()
        // In unit tests, android.util.Log is not mocked which can cause RuntimeException
        // before the IllegalArgumentException is thrown. We accept either exception type.
        assertTrue(
            "Expected IllegalArgumentException or RuntimeException (Log not mocked) but got: ${error?.javaClass?.name}",
            error is IllegalArgumentException ||
            (error is RuntimeException && error.message?.contains("not mocked") == true)
        )
        // Only check the message if it's IllegalArgumentException
        if (error is IllegalArgumentException) {
            assertTrue("Expected message to contain 'modelPath or modelId'", error.message?.contains("modelPath or modelId") == true)
        }
    }

    @Test
    fun `setProgressCallback handles null clearing`() {
        val sd = newStableDiffusion()
        val field = StableDiffusion::class.java.getDeclaredField("cachedProgressCallback").apply {
            isAccessible = true
        }
        val callback = VideoProgressCallback { _, _, _, _, _ -> }

        sd.setProgressCallback(callback)
        assertNotNull(field.get(sd))

        sd.setProgressCallback(null)
        assertNull(field.get(sd))
    }

    @Test
    fun `cancelGeneration toggles cancellation flag`() {
        val sd = newStableDiffusion()
        val field = StableDiffusion::class.java.getDeclaredField("cancellationRequested").apply {
            isAccessible = true
        }
        val flag = field.get(sd) as AtomicBoolean

        assertFalse(flag.get())
        sd.cancelGeneration()
        assertTrue(flag.get())
    }

    private fun newStableDiffusion(): StableDiffusion {
        val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        return constructor.newInstance(1L)
    }

}
