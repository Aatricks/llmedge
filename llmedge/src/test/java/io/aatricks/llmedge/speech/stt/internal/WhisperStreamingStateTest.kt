package io.aatricks.llmedge.speech.stt.internal

import io.aatricks.llmedge.speech.stt.Whisper
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WhisperStreamingStateTest {

    /** 16 samples per ms at 16 kHz; 8 kHz square wave so the VAD high-pass passes it. */
    private fun voiced(ms: Int) = FloatArray(ms * 16) { i -> if (i % 2 == 0) 0.5f else -0.5f }

    private fun silence(ms: Int) = FloatArray(ms * 16)

    /** Fake transcriber returning one segment spanning the whole window. */
    private class WholeWindowTranscriber : (FloatArray, Whisper.TranscribeParams) -> List<Whisper.TranscriptionSegment> {
        val seenParams = mutableListOf<Whisper.TranscribeParams>()

        override fun invoke(
            samples: FloatArray,
            params: Whisper.TranscribeParams,
        ): List<Whisper.TranscriptionSegment> {
            seenParams += params
            val windowCs = samples.size / 160L // 160 samples per centisecond at 16 kHz
            return listOf(Whisper.TranscriptionSegment(0, 0L, windowCs, "window"))
        }
    }

    private fun params(
        stepMs: Int = 1000,
        lengthMs: Int = 2000,
        keepMs: Int = 200,
        useVad: Boolean = false,
    ) = Whisper.StreamingParams(stepMs = stepMs, lengthMs = lengthMs, keepMs = keepMs, useVad = useVad)

    @Test
    fun `segment timestamps are absolute across chunks and the context overlap is deduplicated`() = runBlocking {
        val fake = WholeWindowTranscriber()
        val state = WhisperStreamingState(fake, params())

        state.feedAudio(voiced(1000))
        val first = state.processNextChunk()
        assertEquals(1, first.size)
        assertEquals(0L, first[0].startTime)
        assertEquals(100L, first[0].endTime)

        state.feedAudio(voiced(1000))
        val second = state.processNextChunk()
        assertEquals(1, second.size)
        // Window = 200ms keep-context + 1000ms new, starting at absolute 800ms.
        assertEquals(80L, second[0].startTime)
        assertEquals(200L, second[0].endTime)
        assertEquals(1, second[0].index)
    }

    @Test
    fun `segments ending inside the already-emitted region are not re-emitted`() = runBlocking {
        val fake =
            object : (FloatArray, Whisper.TranscribeParams) -> List<Whisper.TranscriptionSegment> {
                var call = 0

                override fun invoke(
                    samples: FloatArray,
                    p: Whisper.TranscribeParams,
                ): List<Whisper.TranscriptionSegment> {
                    call++
                    return if (call == 1) {
                        listOf(Whisper.TranscriptionSegment(0, 0L, 100L, "first"))
                    } else {
                        // One segment fully inside the keep-context region (already
                        // emitted last window), one genuinely new.
                        listOf(
                            Whisper.TranscriptionSegment(0, 0L, 15L, "stale"),
                            Whisper.TranscriptionSegment(1, 15L, 120L, "fresh"),
                        )
                    }
                }
            }
        val state = WhisperStreamingState(fake, params())

        state.feedAudio(voiced(1000))
        assertEquals(listOf("first"), state.processNextChunk().map { it.text })

        state.feedAudio(voiced(1000))
        val second = state.processNextChunk()
        // "stale" ends at absolute 80+15=95cs <= 100cs already emitted; dropped.
        assertEquals(listOf("fresh"), second.map { it.text })
        assertEquals(80L + 15L, second[0].startTime)
    }

    @Test
    fun `vad-skipped audio still advances the timeline`() = runBlocking {
        val fake = WholeWindowTranscriber()
        val state = WhisperStreamingState(fake, params(useVad = true))

        state.feedAudio(silence(1000))
        assertTrue(state.processNextChunk().isEmpty())

        state.feedAudio(voiced(1000))
        val segments = state.processNextChunk()
        assertEquals(1, segments.size)
        // The silent second was consumed: this window starts at 1000ms - 200ms keep.
        assertEquals(80L, segments[0].startTime)
    }

    @Test
    fun `streaming windows use no_context and a window-sized audio_ctx`() = runBlocking {
        val fake = WholeWindowTranscriber()
        val state = WhisperStreamingState(fake, params())

        state.feedAudio(voiced(1000))
        state.processNextChunk()

        val seen = fake.seenParams.single()
        assertTrue(seen.noContext)
        // 1000ms window = 50 encoder frames + margin, floored at 128.
        assertEquals(128, seen.audioCtx)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `stepMs larger than lengthMs is rejected`() {
        WhisperStreamingState(WholeWindowTranscriber(), params(stepMs = 15000, lengthMs = 10000))
    }

    @Test
    fun `audio backlog is capped instead of growing without bound`() = runBlocking {
        val fake = WholeWindowTranscriber()
        val state = WhisperStreamingState(fake, params(stepMs = 500, lengthMs = 1000))

        repeat(10) { state.feedAudio(voiced(1000)) }
        // Cap is MAX_BACKLOG_WINDOWS (4) x lengthMs.
        assertTrue(
            "buffered=${state.getBufferedAudioMs()}",
            state.getBufferedAudioMs() <= 4000,
        )
    }
}
