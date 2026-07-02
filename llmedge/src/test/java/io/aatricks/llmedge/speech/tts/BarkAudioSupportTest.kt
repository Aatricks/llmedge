package io.aatricks.llmedge.speech.tts

import io.aatricks.llmedge.speech.tts.internal.BarkAudioSupport
import io.aatricks.llmedge.speech.tts.internal.BarkPcmBuffers
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test

class BarkAudioSupportTest {
    @Test
    fun `short synthesis after long one writes only fresh bytes`() {
        val buffers = BarkPcmBuffers()
        val dir = createTempDir(prefix = "bark-wav")
        dir.deleteOnExit()

        val longWav = File(dir, "long.wav")
        BarkAudioSupport.saveAsWav(FloatArray(1000) { 0.5f }, 24000, longWav.absolutePath, buffers)
        assertEquals(44 + 1000 * 2, longWav.length().toInt())

        // Regression: the reused (larger) pcm buffer used to be written in full,
        // appending stale PCM from the previous synthesis past the declared size.
        val shortWav = File(dir, "short.wav")
        BarkAudioSupport.saveAsWav(FloatArray(10) { 0.5f }, 24000, shortWav.absolutePath, buffers)
        assertEquals(44 + 10 * 2, shortWav.length().toInt())
    }
}
