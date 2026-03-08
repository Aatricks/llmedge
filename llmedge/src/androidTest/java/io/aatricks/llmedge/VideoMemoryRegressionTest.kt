package io.aatricks.llmedge

import android.os.Debug
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlin.math.abs
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VideoMemoryRegressionTest : BaseVideoIntegrationTest() {

    @Test
    fun repeatedGenerationsKeepNativeMemoryStable() = runBlocking {
        val frames = buildFrames(frameCount = 6, width = 256, height = 256)
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
                        prompt = "wan skyline",
                        width = 256,
                        height = 256,
                        videoFrames = 6,
                        steps = 18,
                )

        val deltas = mutableListOf<Long>()
        repeat(5) {
            val before = Debug.getNativeHeapAllocatedSize()
            sd.txt2vid(params)
            val after = Debug.getNativeHeapAllocatedSize()
            deltas += after - before
        }

        val maxDelta = deltas.maxOf { abs(it) }
        val threshold = 32L * 1024 * 1024 // 32 MB
        assertTrue("Native heap delta exceeded threshold: $maxDelta", maxDelta < threshold)
    }
}
