package io.aatricks.llmedge.speech.tts

import io.aatricks.llmedge.speech.tts.internal.BarkAudioSupport
import io.aatricks.llmedge.speech.tts.internal.BarkPcmBuffers
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.Future
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
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

    @Test
    fun `concurrent saveAsWav writes distinct valid WAVs`() {
        val buffers = BarkPcmBuffers()
        val dir = createTempDir(prefix = "bark-wav-concurrent")
        dir.deleteOnExit()

        val executor = Executors.newFixedThreadPool(4)
        val futures = mutableListOf<Future<*>>()

        val count = 20
        val sampleSizes = IntArray(count) { i -> 100 + i * 10 }
        val testFiles = Array(count) { i -> File(dir, "concurrent_$i.wav") }

        for (i in 0 until count) {
            val samples = FloatArray(sampleSizes[i]) { if (i % 2 == 0) 0.5f else -0.5f }
            val file = testFiles[i]
            futures.add(executor.submit {
                BarkAudioSupport.saveAsWav(samples, 24000, file.absolutePath, buffers)
            })
        }

        for (future in futures) {
            future.get()
        }
        executor.shutdown()

        // Verify each file is valid and contains non-interleaved data
        for (i in 0 until count) {
            val file = testFiles[i]
            assertTrue(file.exists())
            assertEquals(44 + sampleSizes[i] * 2, file.length().toInt())

            val bytes = file.readBytes()
            val dataBytes = bytes.sliceArray(44 until bytes.size)
            val buffer = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
            val shortBuffer = buffer.asShortBuffer()

            val expectedShort = if (i % 2 == 0) {
                (0.5f * 32767.0f).toInt().toShort()
            } else {
                (-0.5f * 32767.0f).toInt().toShort()
            }

            assertEquals(sampleSizes[i], shortBuffer.remaining())
            while (shortBuffer.hasRemaining()) {
                assertEquals(expectedShort, shortBuffer.get())
            }
        }
    }
}
