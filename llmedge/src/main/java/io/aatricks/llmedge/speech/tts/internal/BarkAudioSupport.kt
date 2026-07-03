package io.aatricks.llmedge.speech.tts.internal

import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.speech.SpeechThreadingSupport
import io.aatricks.llmedge.speech.tts.BarkTTS
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal class BarkPcmBuffers {
    val wavHeaderBuffer: ByteBuffer = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
}

internal object BarkAudioSupport {
    private val WAV_RIFF = "RIFF".toByteArray()
    private val WAV_WAVE = "WAVE".toByteArray()
    private val WAV_FMT = "fmt ".toByteArray()
    private val WAV_DATA = "data".toByteArray()

    fun generate(
        text: String,
        params: BarkTTS.GenerateParams,
        nativeGenerate: (text: String, nThreads: Int) -> FloatArray?,
        nativeGetSampleRate: () -> Int,
    ): BarkTTS.AudioResult {
        require(text.isNotEmpty()) { "Text cannot be empty" }

        val effectiveThreads = SpeechThreadingSupport.resolveThreadCount(params.nThreads)
        val samples =
            nativeGenerate(text, effectiveThreads)
                ?: throw InferenceFailedException(
                    operation = "Bark audio generation",
                    detail = "The native Bark runtime returned no audio samples.",
                )

        val sampleRate = nativeGetSampleRate()
        val durationSeconds = samples.size.toFloat() / sampleRate
        return BarkTTS.AudioResult(samples, sampleRate, durationSeconds)
    }

    fun saveAsWav(
        samples: FloatArray,
        sampleRate: Int,
        filePath: String,
        buffers: BarkPcmBuffers,
    ) {
        val file = File(filePath)
        file.parentFile?.mkdirs()

        synchronized(buffers) {
            FileOutputStream(file).use { fos ->
                fos.write(createWavHeader(samples.size, sampleRate, buffers.wavHeaderBuffer))

                val shortSamples = ShortArray(samples.size)
                for (i in samples.indices) {
                    val clamped = samples[i].coerceIn(-1.0f, 1.0f)
                    shortSamples[i] = (clamped * 32767.0f).toInt().toShort()
                }

                val requiredBytes = samples.size * 2
                val buffer = ByteBuffer.allocate(requiredBytes).order(ByteOrder.LITTLE_ENDIAN)
                buffer.asShortBuffer().put(shortSamples)

                fos.write(buffer.array(), 0, requiredBytes)
            }
        }
    }

    private fun createWavHeader(
        numSamples: Int,
        sampleRate: Int,
        buffer: ByteBuffer,
    ): ByteArray {
        val byteRate = sampleRate * 2
        val dataSize = numSamples * 2
        val fileSize = 36 + dataSize

        buffer.clear()
        buffer.put(WAV_RIFF)
        buffer.putInt(fileSize)
        buffer.put(WAV_WAVE)
        buffer.put(WAV_FMT)
        buffer.putInt(16)
        buffer.putShort(1)
        buffer.putShort(1)
        buffer.putInt(sampleRate)
        buffer.putInt(byteRate)
        buffer.putShort(2)
        buffer.putShort(16)
        buffer.put(WAV_DATA)
        buffer.putInt(dataSize)

        return buffer.array().copyOf()
    }
}
