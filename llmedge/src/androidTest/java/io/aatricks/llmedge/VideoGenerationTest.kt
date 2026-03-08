package io.aatricks.llmedge

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VideoGenerationTest : BaseVideoIntegrationTest() {

    @Test
    fun txt2vidGeneratesBitmapsFromStubBridge() = runBlocking {
        val frames = buildFrames(frameCount = 4, width = 256, height = 256)
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
                ): Array<ByteArray>? = frames.map { it.copyOf() }.toTypedArray()

                override fun setProgressCallback(
                        handle: Long,
                        callback: StableDiffusion.VideoProgressCallback?
                ) = Unit

                override fun cancelGeneration(handle: Long) = Unit
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
        val sd = createStableDiffusion()
        val params =
                StableDiffusion.VideoGenerateParams(
                        prompt = "wan cat walking",
                        width = 256,
                        height = 256,
                        videoFrames = 4,
                        steps = 12,
                        cfgScale = 7.5f,
                )

        val result = sd.txt2vid(params)

        assertEquals(4, result.size)
        assertEquals(256, result.first().width)
        assertTrue(sd.isVideoModel())
        assertNotNull(sd.getLastGenerationMetrics())
    }
}
