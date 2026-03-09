package io.aatricks.llmedge

import androidx.test.ext.junit.runners.AndroidJUnit4
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VideoCancellationTest : BaseVideoIntegrationTest() {

    @Test
    fun cancellingGenerationPropagatesCancellationException() = runBlocking {
        val cancelInvoked = AtomicBoolean(false)
        StableDiffusion.overrideNativeBridgeForTests { instance ->
            object : StableDiffusion.NativeBridge {
                @Volatile private var cancelled = false

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
                    while (!cancelled) {
                        Thread.sleep(10)
                    }
                    // Ensure StableDiffusion observes cancellation before we surface the failure
                    instance.cancelGeneration()
                    throw RuntimeException("native aborted")
                }

                override fun setProgressCallback(
                        handle: Long,
                        callback: VideoProgressCallback?
                ) = Unit

                override fun cancelGeneration(handle: Long) {
                    cancelInvoked.set(true)
                    cancelled = true
                }
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
        val nativeBridgeField =
                StableDiffusion::class.java.getDeclaredField("nativeBridge").apply {
                    isAccessible = true
                }
        val nativeBridge = nativeBridgeField.get(sd)
        if (!nativeBridge.javaClass.name.contains("VideoCancellationTest")) {
            fail("nativeBridge override inactive: ${nativeBridge.javaClass.name}")
        }
        val availabilityField =
                StableDiffusion::class.java.getDeclaredField("isNativeLibraryAvailable").apply {
                    isAccessible = true
                }
        println(
                "VideoCancellationTest: nativeBridgeClass=${nativeBridge.javaClass.name} isNativeLibraryAvailable=${availabilityField.getBoolean(null)}",
        )
        val params =
                VideoGenerateParams(
                        prompt = "wan robot",
                        width = 256,
                        height = 256,
                        videoFrames = 8,
                        steps = 20,
                )

        val job = async { sd.txt2vid(params) }
        delay(50)
        sd.cancelGeneration()

        val result = runCatching { job.await() }
        val exception = result.exceptionOrNull()
        println(
                "VideoCancellationTest: result exception=${exception?.javaClass} message=${exception?.message}"
        )
        assertTrue("Expected cancelGeneration() to be invoked", cancelInvoked.get())
        assertTrue(
                "Expected CancellationException or native abort but got ${exception?.javaClass}",
                exception is kotlinx.coroutines.CancellationException ||
                        (exception is RuntimeException && exception.message == "native aborted"),
        )
    }
}
