package io.aatricks.llmedge

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.text.runtime.SmolLM
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device regression gate for the runtime-hardening fixes that cannot be
 * verified off-device: real native progress/segment callbacks (the JNI
 * thread-cache state is per shared library and must work across translation
 * units) and standard-UTF-8 prompt delivery to native tokenizers.
 *
 * Model files are read from /data/local/tmp (override via instrumentation args):
 *   llmedge.gate.sd_model_path      (default /data/local/tmp/sdturbo.gguf)
 *   llmedge.gate.whisper_model_path (default /data/local/tmp/ggml-base.en.bin)
 *   llmedge.gate.wav_path           (default /data/local/tmp/jfk.wav)
 *   llmedge.gate.text_model_path    (default /data/local/tmp/smollm135.gguf)
 *
 * Run:
 *   ./gradlew :llmedge:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.RuntimeHardeningDeviceGateTest
 */
@RunWith(AndroidJUnit4::class)
class RuntimeHardeningDeviceGateTest {

    private fun requireFile(argKey: String, default: String): File {
        val path = InstrumentationRegistry.getArguments().getString(argKey) ?: default
        val file = File(path)
        assumeTrue("Missing $argKey file: $path", file.exists())
        return file
    }

    @Test
    fun stableDiffusionProgressCallbackFiresDuringTxt2Img() = runBlocking {
        val model = requireFile("llmedge.gate.sd_model_path", "/data/local/tmp/sdturbo.gguf")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val progressCalls = AtomicInteger(0)

        // Pass llmedge.gate.sd_vulkan=true to exercise warm-context reuse on the
        // Vulkan backend (where the original second-generation crash was seen).
        val useVulkan = InstrumentationRegistry.getArguments().getString("llmedge.gate.sd_vulkan") == "true"
        val sd = StableDiffusion.load(context = context, modelPath = model.absolutePath, forceVulkan = useVulkan)
        sd.use { engine ->
            engine.setProgressCallback { _, _, _, _, _ -> progressCalls.incrementAndGet() }
            // Two generations on the same instance: warm sd_ctx reuse crashed on
            // device before 863491a and was untested for every runtime.
            repeat(2) { run ->
                val bitmap = engine.txt2img(
                    GenerateParams(
                        prompt = "a red apple on a table",
                        width = 128,
                        height = 128,
                        steps = 2,
                        cfgScale = 1.0f,
                        seed = 42L + run,
                    ),
                )
                assertTrue("run $run: txt2img returned empty bitmap", bitmap.width > 0 && bitmap.height > 0)
            }
        }
        assertTrue(
            "Native progress callback never fired (JNI thread-cache regression)",
            progressCalls.get() > 0,
        )
    }

    @Test
    fun whisperSegmentAndProgressCallbacksFireDuringTranscribe() {
        val model = requireFile("llmedge.gate.whisper_model_path", "/data/local/tmp/ggml-base.en.bin")
        val wav = requireFile("llmedge.gate.wav_path", "/data/local/tmp/jfk.wav")
        val segmentCalls = AtomicInteger(0)
        val progressCalls = AtomicInteger(0)

        val whisper = Whisper.load(modelPath = model.absolutePath, useGpu = false)
        try {
            whisper.setSegmentCallback { _, _, _, _ -> segmentCalls.incrementAndGet() }
            whisper.setProgressCallback { progressCalls.incrementAndGet() }
            val samples = readWavMono16k(wav)
            repeat(2) { run ->
                val segments = whisper.transcribe(samples)
                assertTrue("run $run: transcription produced no segments", segments.isNotEmpty())
                val text = segments.joinToString(" ") { it.text }.lowercase()
                assertTrue("run $run: unexpected transcript: $text", "country" in text)
            }
        } finally {
            whisper.close()
        }
        assertTrue(
            "Native segment callback never fired (JNI thread-cache regression)",
            segmentCalls.get() > 0,
        )
    }

    @Test
    fun smolLmAcceptsSupplementaryUnicodePrompt() = runBlocking {
        val model = requireFile("llmedge.gate.text_model_path", "/data/local/tmp/smollm135.gguf")
        val smol = SmolLM()
        try {
            smol.load(model.absolutePath, SmolLM.InferenceParams(storeChats = false))
            // U+1F600 crosses JNI as a surrogate pair; before the UTF-8 fix the
            // tokenizer received invalid Modified-UTF-8 bytes for it.
            repeat(2) { run ->
                val response = smol.getResponse("Repeat this exactly: hello 😀 world", maxTokens = 48)
                assertTrue("run $run: empty response to emoji prompt", response.isNotBlank())
            }
        } finally {
            smol.close()
        }
    }

    @Test
    fun smolLmFlowCancellationAndStopCompletionEndGeneration() = runBlocking {
        val model = requireFile("llmedge.gate.text_model_path", "/data/local/tmp/smollm135.gguf")
        val smol = SmolLM()
        try {
            smol.load(model.absolutePath, SmolLM.InferenceParams(storeChats = false))

            // Cancelling the flow (take) must end the stream cleanly.
            val taken = withTimeout(120_000) {
                smol.getResponseAsFlow("Write a very long story about the sea.").take(5).toList()
            }
            assertTrue("take(5) returned ${taken.size} pieces", taken.size == 5)

            // stopCompletion() from the collector must end an in-flight generation.
            val pieces = mutableListOf<String>()
            withTimeout(120_000) {
                smol.getResponseAsFlow("Write another very long story about the sea.").collect {
                    pieces.add(it)
                    if (pieces.size == 5) smol.stopCompletion()
                }
            }
            assertTrue("Generation did not stop promptly (got ${pieces.size} pieces)", pieces.size in 5..40)
        } finally {
            smol.close()
        }
    }

    @Test
    fun barkGeneratesAudioWithGgmlThreadpool() {
        // Bark lost its OpenMP runtime in the libomp-coexistence fix (#39); this
        // confirms generation still works on the ggml threadpool. Opt-in only:
        // bark.cpp needs 20+ minutes for a two-word phrase on a phone CPU, far
        // too slow for the regular gate.
        assumeTrue(
            "Bark gate is opt-in: pass llmedge.gate.run_bark=true (generation takes 20+ min on device)",
            InstrumentationRegistry.getArguments().getString("llmedge.gate.run_bark") == "true",
        )
        val model = requireFile("llmedge.gate.bark_model_path", "/data/local/tmp/bark_ggml_weights.bin")
        val bark = io.aatricks.llmedge.speech.tts.BarkTTS.load(modelPath = model.absolutePath, seed = 42)
        try {
            val audio = bark.generate("Hello world.")
            assertTrue("Bark produced no samples", audio.samples.isNotEmpty())
            assertTrue("Bark produced silent audio", audio.samples.any { it != 0f })
        } finally {
            bark.close()
        }
    }

    /** Minimal RIFF reader for 16 kHz mono PCM16 wav files (e.g. whisper.cpp samples). */
    private fun readWavMono16k(file: File): FloatArray {
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size > 44 && bytes.decodeToString(0, 4) == "RIFF") { "Not a RIFF wav: $file" }
        var pos = 12
        while (pos + 8 <= bytes.size) {
            val id = bytes.decodeToString(pos, pos + 4)
            val size = buffer.getInt(pos + 4)
            if (id == "data") {
                val samples = FloatArray(size / 2)
                for (i in samples.indices) {
                    samples[i] = buffer.getShort(pos + 8 + i * 2) / 32768.0f
                }
                return samples
            }
            pos += 8 + size + (size and 1)
        }
        throw IllegalArgumentException("No data chunk in $file")
    }
}
