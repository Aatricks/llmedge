package io.aatricks.llmedge

import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.ImageGenerationPhase
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StableDiffusionImageTraceTest {
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
    fun `cancellation while waiting on generation mutex logs waiting phase and clears metrics`() = runBlocking {
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
                ): ByteArray = ByteArray(width * height * 3) { 0x55 }

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

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

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

        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply {
            isAccessible = true
        }.newInstance(1L)

        sd.beginImageRequestTrace(99L)
        sd.traceImagePhase(ImageGenerationPhase.REQUESTED, "test requested")

        sd.state.generationMutex.lock()
        val generationJob: Job =
            launch(Dispatchers.Default) {
                try {
                    sd.txt2img(
                        GenerateParams(
                            prompt = "cancel me",
                            width = 128,
                            height = 128,
                            steps = 1,
                        ),
                    )
                } catch (_: CancellationException) {
                    // Expected for this test.
                }
            }

        try {
            waitForPhase(sd, ImageGenerationPhase.WAITING_GENERATION_MUTEX)
            generationJob.cancel(CancellationException("cancel test"))
            sd.cancelGeneration()
            generationJob.join()

            val phases = sd.getLastImageRequestTraceForTests().map { it.phase }
            assertTrue(phases.contains(ImageGenerationPhase.WAITING_GENERATION_MUTEX))
            assertTrue(phases.contains(ImageGenerationPhase.CANCELLED))
            assertFalse(phases.contains(ImageGenerationPhase.COMPLETED))
            assertNull(sd.getLastGenerationMetrics())
        } finally {
            if (sd.state.generationMutex.isLocked) {
                sd.state.generationMutex.unlock()
            }
            sd.close()
        }
    }

    private fun waitForPhase(
        sd: StableDiffusion,
        phase: ImageGenerationPhase,
    ) {
        repeat(20) {
            val phases = sd.getLastImageRequestTraceForTests().map { it.phase }
            if (phase in phases) {
                return
            }
            Thread.sleep(25)
        }
        throw AssertionError("Phase $phase not observed in trace ${sd.getLastImageRequestTraceForTests()}")
    }
}
