package io.aatricks.llmedge.speech

import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.speech.tts.BarkTTS
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Test
import org.robolectric.annotation.Config

@Config(sdk = [34])
class SpeechRequestModelsTest {

    @Test
    fun `WhisperRuntimeRequest maps to load options`() {
        val runtime = WhisperRuntimeRequest(gpuEnabled = true, flashAttention = false, gpuDevice = 2)

        val loadOptions = runtime.toLoadOptions()

        assertEquals(true, loadOptions.useGpu)
        assertEquals(false, loadOptions.flashAttention)
        assertEquals(2, loadOptions.gpuDevice)
    }

    @Test
    fun `BarkRuntimeRequest maps to load options`() {
        val runtime = BarkRuntimeRequest(seed = 7, temperature = 0.8f, fineTemperature = 0.4f, verbosity = 3)

        val loadOptions = runtime.toLoadOptions()

        assertEquals(7, loadOptions.seed)
        assertEquals(0.8f, loadOptions.temperature)
        assertEquals(0.4f, loadOptions.fineTemperature)
        assertEquals(3, loadOptions.verbosity)
    }

    @Test
    fun `Speech request models expose standardized runtime fields`() {
        val speechToText =
            SpeechToTextRequest(
                audioSamples = floatArrayOf(0.1f, 0.2f),
                model = mockk(),
                params = Whisper.TranscribeParams(),
                runtime = WhisperRuntimeRequest(gpuEnabled = true, flashAttention = true, gpuDevice = 1),
            )
        val synthesis =
            SpeechSynthesisRequest(
                text = "hello",
                model = mockk(),
                params = BarkTTS.GenerateParams(),
                runtime = BarkRuntimeRequest(seed = 4),
            )

        assertEquals(true, speechToText.runtime.gpuEnabled)
        assertEquals(true, speechToText.runtime.flashAttention)
        assertEquals(1, speechToText.runtime.gpuDevice)
        assertEquals(4, synthesis.runtime.seed)
    }
}
