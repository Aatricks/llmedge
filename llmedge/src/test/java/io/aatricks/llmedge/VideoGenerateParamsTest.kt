package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VideoGenerateParamsTest {

    @Test
    fun `validate succeeds for typical params`() {
        val params = VideoGenerateParams(
            prompt = "a cat walking",
            width = 512,
            height = 512,
            videoFrames = 16,
            steps = 20,
            cfgScale = 7.0f,
            strength = 0.8f,
        )

        assertTrue(params.validate().isSuccess)
    }

    @Test
    fun `blank prompt fails validation`() {
        val params = VideoGenerateParams(prompt = " ")

        assertValidationFails(params, "Prompt cannot be blank")
    }

    @Test
    fun `width must be multiple of 64`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            width = 510,
        )

        assertValidationFails(params, "Width must be a multiple of 64")
    }

    @Test
    fun `height must be in supported range`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            height = 128,
        )

        assertValidationFails(params, "Height must be a multiple of 64")
    }

    @Test
    fun `frame count must be between 4 and 64`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            videoFrames = 2,
        )

        assertValidationFails(params, "Frame count must be between 5 and 64")
    }

    @Test
    fun `zero frame count fails validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            videoFrames = 0,
        )

        assertValidationFails(params, "Frame count must be between 5 and 64")
    }

    @Test
    fun `steps must be at least 1`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            steps = 0,
        )

        assertValidationFails(params, "Steps must be between 1 and 50")
    }

    @Test
    fun `cfg scale must stay within allowed range`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            cfgScale = 0.5f,
        )

        assertValidationFails(params, "CFG scale must be between 1.0 and 15.0")
    }

    @Test
    fun `strength must stay within zero to one`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            strength = 1.5f,
        )

        assertValidationFails(params, "Strength must be between 0.0 and 1.0")
    }

    @Test
    fun `seed below negative one fails validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            seed = -5L,
        )

        assertValidationFails(params, "Seed must be -1 or non-negative")
    }

    @Test
    fun `withPrompt returns a copy with new prompt`() {
        val params = VideoGenerateParams(prompt = "first")
        val updated = params.withPrompt("second")

        assertEquals("second", updated.prompt)
        assertEquals(params.copy(prompt = "second"), updated)
    }

    @Test
    fun `default factory populates sane values`() {
        val defaults = VideoGenerateParams.default("hey")

        assertEquals("hey", defaults.prompt)
        assertEquals(512, defaults.width)
        assertEquals(16, defaults.videoFrames)
    }

    // T087: Edge case tests for parameter boundaries
    
    @Test
    fun `maximum resolution 960x960 passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            width = 960,
            height = 960,
        )

        assertTrue("Max resolution should be valid", params.validate().isSuccess)
    }

    @Test
    fun `resolution above 960 fails validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            width = 1024,
            height = 960,
        )

        assertValidationFails(params, "Width must be a multiple of 64 in range 256..960")
    }

    @Test
    fun `maximum 64 frames passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            videoFrames = 64,
        )

        assertTrue("Max frames (64) should be valid", params.validate().isSuccess)
    }

    @Test
    fun `frames above 64 fail validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            videoFrames = 65,
        )

        assertValidationFails(params, "Frame count must be between 5 and 64")
    }

    @Test
    fun `minimum 5 frames passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            videoFrames = 5,
        )

        assertTrue("Min frames (5) should be valid", params.validate().isSuccess)
    }

    @Test
    fun `minimum width 256 passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            width = 256,
            height = 256,
        )

        assertTrue("Min resolution 256x256 should be valid", params.validate().isSuccess)
    }

    @Test
    fun `width below 256 fails validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            width = 192,
        )

        assertValidationFails(params, "Width must be a multiple of 64 in range 256..960")
    }

    @Test
    fun `minimum 1 step passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            steps = 1,
        )

        assertTrue("Min steps (1) should be valid", params.validate().isSuccess)
    }

    @Test
    fun `maximum 50 steps passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            steps = 50,
        )

        assertTrue("Max steps (50) should be valid", params.validate().isSuccess)
    }

    @Test
    fun `minimum cfg scale 1_0 passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            cfgScale = 1.0f,
        )

        assertTrue("Min CFG scale (1.0) should be valid", params.validate().isSuccess)
    }

    @Test
    fun `maximum cfg scale 15_0 passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            cfgScale = 15.0f,
        )

        assertTrue("Max CFG scale (15.0) should be valid", params.validate().isSuccess)
    }

    @Test
    fun `zero strength passes validation for T2V mode`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            strength = 0.0f,
            initImage = null,
        )

        assertTrue("Zero strength should be valid for T2V (no init image)", params.validate().isSuccess)
    }

    @Test
    fun `one strength passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            strength = 1.0f,
        )

        assertTrue("Strength 1.0 should be valid", params.validate().isSuccess)
    }

    // T088: Parameter combination tests
    
    @Test
    fun `high resolution with high frame count passes validation`() {
        // This tests memory-intensive combinations but doesn't test OOM warning
        // (warnings are logged, not validation failures)
        val params = VideoGenerateParams(
            prompt = "valid",
            width = 960,
            height = 960,
            videoFrames = 64,
        )

        assertTrue("High resolution + high frames should pass validation", params.validate().isSuccess)
    }

    @Test
    fun `medium resolution with medium frames passes validation`() {
        val params = VideoGenerateParams(
            prompt = "valid",
            width = 512,
            height = 512,
            videoFrames = 32,
        )

        assertTrue("Medium params should be valid", params.validate().isSuccess)
    }

    @Test
    fun `all schedulers pass validation`() {
        for (scheduler in Scheduler.values()) {
            val params = VideoGenerateParams(
                prompt = "valid",
                scheduler = scheduler,
            )

            assertTrue("Scheduler $scheduler should be valid", params.validate().isSuccess)
        }
    }

    private fun assertValidationFails(
        params: VideoGenerateParams,
        expectedMessagePart: String,
    ) {
        val failure = params.validate().exceptionOrNull()
        assertNotNull("Expected validation failure", failure)
        assertTrue(failure is IllegalArgumentException)
        assertTrue(
            "Error message should mention '$expectedMessagePart' but was '${failure?.message}'",
            failure?.message?.contains(expectedMessagePart) == true,
        )
    }
}
