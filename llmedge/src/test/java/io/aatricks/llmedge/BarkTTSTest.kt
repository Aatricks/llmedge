package io.aatricks.llmedge

import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.speech.tts.BarkTTS

/**
 * Unit tests for BarkTTS class - tests that don't require native library. These test the Kotlin API
 * layer and data classes.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class BarkTTSTest {

    @Test
    fun `EncodingStep fromInt returns correct values`() {
        assertEquals(BarkTTS.EncodingStep.SEMANTIC, BarkTTS.EncodingStep.fromInt(0))
        assertEquals(BarkTTS.EncodingStep.COARSE, BarkTTS.EncodingStep.fromInt(1))
        assertEquals(BarkTTS.EncodingStep.FINE, BarkTTS.EncodingStep.fromInt(2))
        // Unknown values should return SEMANTIC as default
        assertEquals(BarkTTS.EncodingStep.SEMANTIC, BarkTTS.EncodingStep.fromInt(99))
    }

    @Test
    fun `EncodingStep values are correct`() {
        assertEquals(0, BarkTTS.EncodingStep.SEMANTIC.value)
        assertEquals(1, BarkTTS.EncodingStep.COARSE.value)
        assertEquals(2, BarkTTS.EncodingStep.FINE.value)
    }

    @Test
    fun `AudioResult durationMs calculates correctly`() {
        val audio =
                BarkTTS.AudioResult(
                        samples = FloatArray(24000), // 1 second at 24kHz
                        sampleRate = 24000,
                        durationSeconds = 1.0f
                )

        assertEquals(1000L, audio.durationMs)
    }

    @Test
    fun `AudioResult durationMs for longer audio`() {
        val audio =
                BarkTTS.AudioResult(
                        samples = FloatArray(72000), // 3 seconds at 24kHz
                        sampleRate = 24000,
                        durationSeconds = 3.0f
                )

        assertEquals(3000L, audio.durationMs)
    }

    @Test
    fun `AudioResult equality works correctly`() {
        val samples = floatArrayOf(0.1f, 0.2f, 0.3f)
        val audio1 = BarkTTS.AudioResult(samples, 24000, 0.5f)
        val audio2 = BarkTTS.AudioResult(samples, 24000, 0.5f)

        assertEquals(audio1, audio2)
        assertEquals(audio1.hashCode(), audio2.hashCode())
    }

    @Test
    fun `AudioResult inequality with different samples`() {
        val audio1 = BarkTTS.AudioResult(floatArrayOf(0.1f, 0.2f), 24000, 0.5f)
        val audio2 = BarkTTS.AudioResult(floatArrayOf(0.3f, 0.4f), 24000, 0.5f)

        assertNotEquals(audio1, audio2)
    }

    @Test
    fun `AudioResult inequality with different sample rate`() {
        val samples = floatArrayOf(0.1f, 0.2f)
        val audio1 = BarkTTS.AudioResult(samples, 24000, 0.5f)
        val audio2 = BarkTTS.AudioResult(samples, 16000, 0.5f)

        assertNotEquals(audio1, audio2)
    }

    @Test
    fun `GenerateParams default values are correct`() {
        val params = BarkTTS.GenerateParams()

        assertEquals(0, params.nThreads)
    }

    @Test
    fun `GenerateParams custom values`() {
        val params = BarkTTS.GenerateParams(nThreads = 8)

        assertEquals(8, params.nThreads)
    }

    @Test
    fun `SAMPLE_RATE constant is 24000`() {
        assertEquals(24000, BarkTTS.SAMPLE_RATE)
    }

    @Test
    fun `AudioResult with zero duration`() {
        val audio =
                BarkTTS.AudioResult(
                        samples = FloatArray(0),
                        sampleRate = 24000,
                        durationSeconds = 0.0f
                )

        assertEquals(0L, audio.durationMs)
        assertEquals(0, audio.samples.size)
    }

    @Test
    fun `AudioResult with fractional duration`() {
        val audio =
                BarkTTS.AudioResult(
                        samples = FloatArray(12000), // 0.5 seconds at 24kHz
                        sampleRate = 24000,
                        durationSeconds = 0.5f
                )

        assertEquals(500L, audio.durationMs)
    }

    @Test
    fun `EncodingStep entries contains all values`() {
        val entries = BarkTTS.EncodingStep.entries

        assertEquals(3, entries.size)
        assertTrue(entries.contains(BarkTTS.EncodingStep.SEMANTIC))
        assertTrue(entries.contains(BarkTTS.EncodingStep.COARSE))
        assertTrue(entries.contains(BarkTTS.EncodingStep.FINE))
    }
}
