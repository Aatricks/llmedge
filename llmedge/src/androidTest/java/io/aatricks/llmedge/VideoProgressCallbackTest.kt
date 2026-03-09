package io.aatricks.llmedge

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VideoProgressCallbackTest : BaseVideoIntegrationTest() {

    @Test
    fun forwardsProgressEventsFromNativeBridge() = runBlocking {
        val frames = buildFrames(frameCount = 4, width = 256, height = 256)
        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
                private var callback: VideoProgressCallback? = null

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
                ): Array<ByteArray>? {
                    repeat(videoFrames) { index ->
                        callback?.onProgress(
                                step = index + 1,
                                totalSteps = steps,
                                currentFrame = index + 1,
                                totalFrames = videoFrames,
                                timePerStep = 0.05f,
                        )
                    }
                    return frames.map { it.copyOf() }.toTypedArray()
                }

                override fun setProgressCallback(
                        handle: Long,
                        callback: VideoProgressCallback?
                ) {
                    this.callback = callback
                }

                override fun cancelGeneration(handle: Long) = Unit
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
        val sd = createStableDiffusion()
        val progressEvents = mutableListOf<Pair<Int, Int>>()
        val params =
                VideoGenerateParams(
                        prompt = "wan fox in snow",
                        width = 256,
                        height = 256,
                        videoFrames = 4,
                        steps = 16,
                )

        sd.txt2vid(params) { _, _, currentFrame, totalFrames, _ ->
            progressEvents += currentFrame to totalFrames
        }

        assertTrue(progressEvents.isNotEmpty())
        val last = progressEvents.last()
        assertEquals(4, last.first)
        assertEquals(4, last.second)
    }
}
