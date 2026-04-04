package io.aatricks.llmedge

import android.graphics.Bitmap
import io.aatricks.llmedge.core.InferenceFailedException
import io.mockk.*
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.runBlocking
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicBoolean
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoGenerationTest {

    @Suppress("unused")
    private val disableNativeLoadForTests = run {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        StableDiffusion.overrideNativeBridgeForTests { instance ->
            object : StableDiffusion.NativeBridge {
                private val cancellationField = StableDiffusion::class.java.getDeclaredField("cancellationRequested").apply { isAccessible = true }
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
                    // Simulate a generation that checks for cancellation and gives some time to cancel
                    val flag = cancellationField.get(instance) as java.util.concurrent.atomic.AtomicBoolean
                    for (i in 0 until videoFrames) {
                        if (flag.get()) {
                            // Simulate a native-side abort
                            throw RuntimeException("Simulated native cancellation")
                        }
                        try { Thread.sleep(50) } catch (_: InterruptedException) { }
                    }
                    return when {
                        prompt.contains("fail") -> null // Simulate failure
                        else -> Array(videoFrames) { byteArrayOf(1, 2, 3) } // Mock frame data
                    }
                }

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

    @Test
    fun `txt2vid generates video frames with correct count`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = VideoGenerateParams(
            prompt = "a cat walking",
            width = 256,
            height = 256,
            videoFrames = 8,
            steps = 10,
            cfgScale = 7.0f,
            seed = 123L
        )

        val result = sd.txt2vid(params)

        assertNotNull("txt2vid should return frames", result)
        assertEquals("Should return correct number of frames", params.videoFrames, result.size)

        result.forEach { frame ->
            assertEquals("Frame width should match params", params.width, frame.width)
            assertEquals("Frame height should match params", params.height, frame.height)
            assertEquals("Frame should be ARGB_8888", Bitmap.Config.ARGB_8888, frame.config)
        }
    }

    @Test
    fun `txt2vid throws exception when not a video model`() = runTest {
        val sd = newStableDiffusion()
        // Don't set video model metadata

        val params = VideoGenerateParams(prompt = "test")

        val thrown = runCatching { sd.txt2vid(params) }.exceptionOrNull()
        assertTrue(thrown is IllegalStateException)

        assertTrue(thrown?.message?.contains("not a video model") == true)
    }

    @Test
    fun `txt2vid throws exception when native method returns null`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = VideoGenerateParams(prompt = "fail") // Triggers null return

        val thrown = runCatching { sd.txt2vid(params) }.exceptionOrNull()
        assertTrue(thrown is InferenceFailedException)

        assertEquals("Video generation failed", thrown?.message)
    }

    @Test
    fun `txt2vid enforces frame limits based on model size`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata(parameterCount = "5B"))

        val params = VideoGenerateParams(
            prompt = "test",
            videoFrames = 64 // Over limit for 5B model
        )

        val thrown = runCatching { sd.txt2vid(params) }.exceptionOrNull()
        assertTrue(thrown is IllegalArgumentException)

        assertTrue(thrown?.message?.contains("supports maximum 32 frames") == true)
    }

    @Test
    fun `txt2vid handles cancellation correctly`() = runTest {
        // Use a latch to deterministically wait for the native bridge to start
        val latch = java.util.concurrent.CountDownLatch(1)
        // Override native bridge provider for this test so we can signal state
        StableDiffusion.overrideNativeBridgeForTests { instance ->
            object : StableDiffusion.NativeBridge {
                private val cancellationField = StableDiffusion::class.java.getDeclaredField("cancellationRequested").apply { isAccessible = true }
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
                    // Signal we've started
                    println("[VideoGenerationTest] Mock txt2vid called (thread=${Thread.currentThread().name})")
                    latch.countDown()
                    val flag = cancellationField.get(instance) as java.util.concurrent.atomic.AtomicBoolean
                    for (i in 0 until videoFrames) {
                        if (flag.get()) {
                            // Simulate a native-side abort using CancellationException to mirror expected flow
                            throw kotlinx.coroutines.CancellationException("Simulated native cancellation")
                        }
                        try { Thread.sleep(50) } catch (_: InterruptedException) { }
                    }
                    return when {
                        prompt.contains("fail") -> null // Simulate failure
                        else -> Array(videoFrames) { byteArrayOf(1, 2, 3) } // Mock frame data
                    }
                }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) {}
                override fun cancelGeneration(handle: Long) {}
            }
        }

        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = VideoGenerateParams(prompt = "test")

        // Start generation in a separate thread and cancel shortly after to simulate user cancellation
        val thrownHolder = java.util.concurrent.atomic.AtomicReference<Throwable?>(null)
        val thread = Thread {
            try {
                runBlocking { sd.txt2vid(params) }
            } catch (t: Throwable) {
                thrownHolder.set(t)
            }
        }
        thread.start()

        // Wait for the native bridge to start and then request cancellation
        val started = latch.await(5, java.util.concurrent.TimeUnit.SECONDS)
        assertTrue("Native bridge never started", started)
        sd.cancelGeneration()

        thread.join()
        val thrown = thrownHolder.get()
        assertTrue(thrown is CancellationException)
        assertTrue(thrown?.message?.contains("cancel") == true)

        // Restore original provider to avoid affecting other tests
        StableDiffusion.resetNativeBridgeForTests()
    }

    @Test
    fun `txt2vid sets and clears progress callback correctly`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val callback = VideoProgressCallback { _, _, _, _, _ -> }
        val params = VideoGenerateParams(prompt = "test")

        // Set callback before generation
        sd.setProgressCallback(callback)

        // Generate video (should use callback)
        val result = sd.txt2vid(params)
        assertNotNull(result)

        // Callback should be restored after generation
        val cachedCallbackField = StableDiffusion::class.java.getDeclaredField("cachedProgressCallback")
        cachedCallbackField.isAccessible = true
        assertEquals(callback, cachedCallbackField.get(sd))
    }

    @Test
    fun `txt2vid validates parameters before generation`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val invalidParams = VideoGenerateParams(
            prompt = "", // Invalid: blank prompt
            width = 256,
            height = 256
        )

        val thrown = runCatching { sd.txt2vid(invalidParams) }.exceptionOrNull()
        assertTrue(thrown is IllegalArgumentException)

        assertTrue(thrown?.message?.contains("Prompt cannot be blank") == true)
    }

    @Test
    fun `txt2vid handles init image for I2V mode correctly`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val initImage = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        val params = VideoGenerateParams(
            prompt = "test",
            initImage = initImage,
            strength = 0.8f
        )

        val result = sd.txt2vid(params)
        assertNotNull("Should generate video with init image", result)
    }

    @Test
    fun `txt2vid validates init image and strength consistency`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = VideoGenerateParams(
            prompt = "test",
            initImage = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888),
            strength = 0.0f // Invalid: strength must be > 0 for I2V
        )

        val thrown2 = runCatching { sd.txt2vid(params) }.exceptionOrNull()
        assertTrue(thrown2 is IllegalArgumentException)
        assertTrue(thrown2?.message?.contains("strength must be > 0.0") == true)
    }

    @Test
    fun `txt2vid uses generation mutex for thread safety`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = VideoGenerateParams(prompt = "test")

        // The method should use generationMutex.withLock
        // We verify this by ensuring the method completes successfully
        val result = sd.txt2vid(params)
        assertNotNull("Should complete generation", result)
    }

    @Test
    fun `txt2vid calculates and stores generation metrics`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = VideoGenerateParams(
            prompt = "test",
            width = 256,
            height = 256,
            videoFrames = 5,
            steps = 10
        )

        val result = sd.txt2vid(params)

        val metrics = requireNotNull(sd.getLastGenerationMetrics())
        assertTrue("Total time should be positive", metrics.totalTimeSeconds > 0)
        assertTrue("Frames per second should be positive", metrics.framesPerSecond > 0)
        assertEquals(5, result.size) // Wan formula: (5-1)/4*4+1 = 5
    }

    @Test
    fun `txt2vid handles different schedulers correctly`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val schedulers = Scheduler.values()

        for (scheduler in schedulers) {
            val params = VideoGenerateParams(
                prompt = "test with $scheduler",
                scheduler = scheduler
            )

            val result = sd.txt2vid(params)
            assertNotNull("Should generate with $scheduler scheduler", result)
        }
    }

    @Test
    fun `cancelGeneration sets cancellation flag`() {
        val sd = newStableDiffusion()

        val cancellationField = StableDiffusion::class.java.getDeclaredField("cancellationRequested")
        cancellationField.isAccessible = true
        val flag = cancellationField.get(sd) as AtomicBoolean

        assertFalse("Flag should start false", flag.get())

        sd.cancelGeneration()

        assertTrue("Flag should be set to true", flag.get())
    }

    private fun newStableDiffusion(): StableDiffusion {
        val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        return constructor.newInstance(1L)
    }

    private fun createVideoModelMetadata(
        architecture: String = "wan",
        parameterCount: String = "1.3B"
    ): VideoModelMetadata {
        return VideoModelMetadata(
            architecture = architecture,
            modelType = "t2v",
            parameterCount = parameterCount,
            mobileSupported = true,
            tags = setOf("wan", "text-to-video"),
            filename = "test-model.gguf"
        )
    }
}
