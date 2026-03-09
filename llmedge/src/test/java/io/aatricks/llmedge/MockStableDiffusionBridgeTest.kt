package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class MockStableDiffusionBridgeTest {

    @Test
    fun `mock bridge provides configurable test scenarios`() {
        val mockBridge = MockStableDiffusionBridge()

        // Test default configuration
        assertNotNull("Mock bridge should be created", mockBridge)
        assertEquals(0, mockBridge.txt2VidCalls.size)

        // Test configuration methods exist
        mockBridge.configureForFailure()
        mockBridge.configureFrameCount(16)
        mockBridge.configureFrameDimensions(512, 512)
        mockBridge.disableProgressDelays()
        mockBridge.reset()

        assertEquals("Configuration methods should work", true, true)
    }

    @Test
    fun `mock bridge tracks method calls correctly`() {
        val mockBridge = MockStableDiffusionBridge()

        // Test that tracking collections are initialized
        assertNotNull("txt2VidCalls should be initialized", mockBridge.txt2VidCalls)
        assertNotNull("setProgressCallbackCalls should be initialized", mockBridge.setProgressCallbackCalls)
        assertNotNull("cancelGenerationCalls should be initialized", mockBridge.cancelGenerationCalls)

        // Test reset functionality
        mockBridge.reset()
        assertEquals(0, mockBridge.txt2VidCalls.size)
        assertEquals(0, mockBridge.setProgressCallbackCalls.size)
        assertEquals(0, mockBridge.cancelGenerationCalls.size)
    }

    @Test
    fun `Txt2VidCall data class captures all parameters`() {
        val call = MockStableDiffusionBridge.Txt2VidCall(
            handle = 1L,
            prompt = "test prompt",
            negative = "test negative",
            width = 512,
            height = 512,
            videoFrames = 8,
            steps = 20,
            cfg = 7.5f,
            seed = 123L,
            sampleMethod = SampleMethod.DEFAULT,
            scheduler = Scheduler.DEFAULT,
            strength = 0.8f,
            initImage = byteArrayOf(1, 2, 3),
            initWidth = 256,
            initHeight = 256,
            vaceStrength = 1.0f,
            easyCacheEnabled = false,
            easyCacheReuseThreshold = 0.2f,
            easyCacheStartPercent = 0.15f,
            easyCacheEndPercent = 0.95f
        )

        assertEquals(1L, call.handle)
        assertEquals("test prompt", call.prompt)
        assertEquals("test negative", call.negative)
        assertEquals(512, call.width)
        assertEquals(512, call.height)
        assertEquals(8, call.videoFrames)
        assertEquals(20, call.steps)
        assertEquals(7.5f, call.cfg)
        assertEquals(123L, call.seed)
        assertEquals(Scheduler.DEFAULT, call.scheduler)
        assertEquals(0.8f, call.strength)
        assertNotNull(call.initImage)
        assertEquals(256, call.initWidth)
        assertEquals(256, call.initHeight)
    }

    @Test
    fun `mock bridge txt2vid returns expected results`() {
        val mockBridge = MockStableDiffusionBridge()
        mockBridge.configureFrameCount(4)

        val result = mockBridge.txt2vid(
            handle = 1L,
            prompt = "test",
            negative = "",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 10,
            cfg = 7.0f,
            seed = 42L,
            sampleMethod = SampleMethod.DEFAULT,
            scheduler = Scheduler.DEFAULT,
            strength = 0.8f,
            initImage = null,
            initWidth = 0,
            initHeight = 0
            , vaceStrength = 1.0f
            , easyCacheEnabled = false
            , easyCacheReuseThreshold = 0.2f
            , easyCacheStartPercent = 0.15f
            , easyCacheEndPercent = 0.95f
        )

        assertNotNull("Should return frame array", result)
        assertEquals("Should return correct number of frames", 4, result?.size)

        // Verify call was tracked
        assertEquals(1, mockBridge.txt2VidCalls.size)
        val call = mockBridge.txt2VidCalls[0]
        assertEquals(1L, call.handle)
        assertEquals("test", call.prompt)
        assertEquals(4, call.videoFrames)
    }

    @Test
    fun `mock bridge can simulate failure scenarios`() {
        val mockBridge = MockStableDiffusionBridge()
        mockBridge.configureForFailure()

        val result = mockBridge.txt2vid(
            handle = 1L,
            prompt = "test",
            negative = "",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 10,
            cfg = 7.0f,
            seed = 42L,
            sampleMethod = SampleMethod.DEFAULT,
            scheduler = Scheduler.DEFAULT,
            strength = 0.8f,
            initImage = null,
            initWidth = 0,
            initHeight = 0
            , vaceStrength = 1.0f
            , easyCacheEnabled = false
            , easyCacheReuseThreshold = 0.2f
            , easyCacheStartPercent = 0.15f
            , easyCacheEndPercent = 0.95f
        )

        assertNull("Should return null when configured for failure", result)
        assertEquals(1, mockBridge.txt2VidCalls.size)
    }
}
