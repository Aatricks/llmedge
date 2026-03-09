package io.aatricks.llmedge

import android.graphics.Bitmap
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import io.mockk.*
import org.junit.Test
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import kotlinx.coroutines.delay

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StableDiffusionIntegrationTest {

    /**
     * Small helper to run tests with the JVM-level native-load disabled and the
     * StableDiffusion native bridge enabled; guarantees reset and property cleanup.
     */
    private suspend inline fun <R> withStableDiffusionNativeBridgeDisabledSuspend(crossinline block: suspend () -> R): R {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        try {
            return block()
        } finally {
            StableDiffusion.resetNativeBridgeForTests()
            System.clearProperty("llmedge.disableNativeLoad")
        }
    }

    private inline fun <R> withStableDiffusionNativeBridgeDisabled(crossinline block: () -> R): R {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        try {
            return block()
        } finally {
            StableDiffusion.resetNativeBridgeForTests()
            System.clearProperty("llmedge.disableNativeLoad")
        }
    }

    @Test
    fun `txt2vid uses native bridge and provides progress callbacks`() = runTest {
        withStableDiffusionNativeBridgeDisabledSuspend {
        // Disable native load for JVM tests and inject mock bridge
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()

        val mockBridge = MockStableDiffusionBridge()
        // Disable delays for deterministic and fast tests
        mockBridge.disableProgressDelays()

        StableDiffusion.overrideNativeBridgeForTests { _ -> mockBridge }

        // Using Robolectric to create real Bitmaps for JVM tests, so no mocking required here.

        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val progressCounter = AtomicInteger(0)
        val callback = VideoProgressCallback { _, _, _, _, _ -> progressCounter.incrementAndGet() }

        // Set the cached progress callback and verify the bridge receives setProgressCallback call
        sd.setProgressCallback(callback)
        assertTrue("bridge should receive a setProgressCallback call", mockBridge.setProgressCallbackCalls.isNotEmpty())
        assertEquals(callback, mockBridge.setProgressCallbackCalls.last().second)

        val params = VideoGenerateParams(
            prompt = "integration test prompt",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 10,
            cfgScale = 7.0f,
            seed = 123L
        )

        val bitmaps = sd.txt2vid(params)

        // Verify bridge got the native call and correct params
        assertEquals(1, mockBridge.txt2VidCalls.size)
        val call = mockBridge.txt2VidCalls.first()
        assertEquals(params.prompt, call.prompt)
        assertEquals(params.videoFrames, call.videoFrames)
        assertEquals(params.width, call.width)
        assertEquals(params.height, call.height)

        // Verify progress callback was invoked at least once (per step)
        assertTrue("Progress callback should be invoked", progressCounter.get() >= 1)

        // Verify bitmap count and dimensions match params
        assertEquals(params.videoFrames, bitmaps.size)
        bitmaps.forEach { bmp ->
            assertEquals(params.width, bmp.width)
            assertEquals(params.height, bmp.height)
            assertEquals(Bitmap.Config.ARGB_8888, bmp.config)
        }

        // Verify conversion of first pixel matches expected pattern from MockStableDiffusionBridge
        val pixel = bitmaps[0].getPixel(0, 0)
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        assertEquals(0, r)
        assertEquals(13, g)
        assertEquals(26, b)

        }
    }

    @Test
    fun `txt2vid onProgress argument overrides cached progress callback and is restored`() = runTest {
        withStableDiffusionNativeBridgeDisabledSuspend {
        val mockBridge = MockStableDiffusionBridge()
        mockBridge.disableProgressDelays()
        StableDiffusion.overrideNativeBridgeForTests { _ -> mockBridge }

        // Using Robolectric to create real Bitmaps for JVM tests, so no mocking required here.

        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val cachedCounter = AtomicInteger(0)
        val cachedCb = VideoProgressCallback { _, _, _, _, _ -> cachedCounter.incrementAndGet() }
        sd.setProgressCallback(cachedCb)

        val onProgressCounter = AtomicInteger(0)
        val onProgressCb = VideoProgressCallback { _, _, _, _, _ -> onProgressCounter.incrementAndGet() }

        val params = VideoGenerateParams(
            prompt = "test override callback",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 10,
        )

        // Clear previous bridge calls for easier assertions
        mockBridge.setProgressCallbackCalls.clear()
        mockBridge.txt2VidCalls.clear()

        val bitmaps = sd.txt2vid(params, onProgress = onProgressCb)

        // Verify onProgress callback got invoked, and cached callback did not (or was restored)
        assertTrue(onProgressCounter.get() >= 1)
        // At the end, the bridge should have been restored to the cached callback
        assertTrue(mockBridge.setProgressCallbackCalls.size >= 2)
        val lastCall = mockBridge.setProgressCallbackCalls.last()
        assertEquals(cachedCb, lastCall.second)

        // Confirm bitmaps returned as expected
        assertEquals(params.videoFrames, bitmaps.size)

        }
    }

    @Test
    fun `txt2vid cancellation during generation triggers CancellationException`() {
        withStableDiffusionNativeBridgeDisabled {
        val mockBridge = MockStableDiffusionBridge()
        // Use progress delays to make generation long enough to cancel
        mockBridge.progressCallbackDelayMs = 30L
        // Larger steps to increase generation time
        val steps = 20
        StableDiffusion.overrideNativeBridgeForTests { _ -> mockBridge }

        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = VideoGenerateParams(
            prompt = "cancel test",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = steps
        )

        val exceptionRef = AtomicReference<Throwable?>(null)
        val progressCounter = AtomicInteger(0)
        val cancelingCb = VideoProgressCallback { _, _, _, _, _ ->
            if (progressCounter.incrementAndGet() == 1) {
                // Cancel on first progress update to ensure mid-run cancellation
                sd.cancelGeneration()
            }
        }

        val t = thread(start = true) {
            try {
                runBlocking { sd.txt2vid(params, onProgress = cancelingCb) }
            } catch (e: Throwable) {
                exceptionRef.set(e)
            }
        }

        t.join(2000)

        val thrown = exceptionRef.get()
        assertNotNull("Generation should have thrown an exception due to cancellation", thrown)
        assertTrue(
            "Expected CancellationException or wrapped cause",
            thrown is kotlinx.coroutines.CancellationException || thrown?.cause is kotlinx.coroutines.CancellationException
        )

        }
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
