package io.aatricks.llmedge

import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback

class StableDiffusionVideoModelDetectorExtraTest {
    @Before
    fun setUp() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        StableDiffusion.overrideNativeBridgeForTests {
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
    }

    @After
    fun tearDown() {
        StableDiffusion.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `filename containing WAR casing doesn't prevent detection`() {
        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = null,
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = emptySet(),
                filename = "HUNYUAN_video_model.gguf",
            )
        )
        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `tags case-insensitive detection`() {
        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "stable-diffusion-xl",
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = setOf("Text-to-Video"),
                filename = "sdxl.gguf",
            ),
        )
        assertTrue(sd.isVideoModel())
    }
}
