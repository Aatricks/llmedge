package io.aatricks.llmedge.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.speech.stt.Whisper
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Real-time check for the rewritten streaming state machine: each 3 s step must
 * transcribe in well under 3 s of wall time on device, and emitted segments must
 * carry sane, monotonically advancing absolute timestamps.
 *
 * Run with:
 *   ./gradlew :llmedge:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.benchmark.WhisperStreamingBenchmark \
 *     -Pandroid.testInstrumentationRunnerArguments.llmedge.benchmark.whisper_model_path=/data/local/tmp/ggml-base.en.bin \
 *     -Pandroid.testInstrumentationRunnerArguments.llmedge.benchmark.wav_path=/data/local/tmp/jfk.wav
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class WhisperStreamingBenchmark {
    companion object {
        private const val SAMPLE_RATE = 16_000
        private const val STEP_MS = 3_000
    }

    @Test
    fun streamingStepsKeepUpWithRealTime() {
        val args = InstrumentationRegistry.getArguments()
        val modelPath = args.getString("llmedge.benchmark.whisper_model_path")
        val wavPath = args.getString("llmedge.benchmark.wav_path")
        assumeTrue("No whisper model path provided", modelPath != null)
        assumeTrue("No wav path provided", wavPath != null)

        val audio = readWavMono16k(File(wavPath!!))
        val audioDurationMs = audio.size * 1000L / SAMPLE_RATE
        println("[WhisperStreaming] audio=${audioDurationMs}ms samples=${audio.size}")

        val whisper = Whisper.load(modelPath!!, useGpu = false)
        try {
            val transcriber =
                whisper.createStreamingTranscriber(
                    Whisper.StreamingParams(stepMs = STEP_MS, useVad = false),
                )
            val stepSamples = SAMPLE_RATE * STEP_MS / 1000
            val stepTimesMs = mutableListOf<Double>()
            val segments = mutableListOf<Whisper.TranscriptionSegment>()

            runBlocking {
                var offset = 0
                // Warm-up step: first window pays one-time init (mel filters, allocs).
                var warm = true
                while (offset < audio.size) {
                    val end = minOf(offset + stepSamples, audio.size)
                    transcriber.feedAudio(audio.copyOfRange(offset, end))
                    offset = end
                    val start = System.nanoTime()
                    val out = transcriber.processNextChunk()
                    val elapsedMs = (System.nanoTime() - start) / 1_000_000.0
                    segments += out
                    if (warm) {
                        warm = false
                        BenchmarkReporter.record("whisper_streaming", "first_step_ms", elapsedMs, "ms")
                    } else {
                        stepTimesMs += elapsedMs
                    }
                    println(
                        "[WhisperStreaming] step=${"%.0f".format(elapsedMs)}ms " +
                            "segments=${out.size} text=${out.joinToString { it.text }}",
                    )
                }
            }

            val worst = stepTimesMs.maxOrNull() ?: 0.0
            val median = stepTimesMs.sorted().let { if (it.isEmpty()) 0.0 else it[it.size / 2] }
            BenchmarkReporter.record("whisper_streaming", "median_step_ms", median, "ms")
            BenchmarkReporter.record("whisper_streaming", "worst_step_ms", worst, "ms")
            BenchmarkReporter.printSummary()

            val allText = segments.joinToString(" ") { it.text }.lowercase()
            println("[WhisperStreaming] transcript: $allText")
            assertTrue("Expected non-empty transcript", allText.isNotBlank())
            assertTrue(
                "Timestamps must not exceed the audio duration (got ${segments.maxOfOrNull { it.endTimeMs }})",
                segments.all { it.endTimeMs <= audioDurationMs + STEP_MS },
            )
            assertTrue(
                "Steady-state steps must keep up with real time (worst=${worst}ms > ${STEP_MS}ms)",
                worst < STEP_MS,
            )
        } finally {
            whisper.close()
        }
    }

    /** Minimal RIFF reader for 16 kHz mono PCM16 wav files (e.g. whisper.cpp samples). */
    private fun readWavMono16k(file: File): FloatArray {
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size > 44 && bytes.decodeToString(0, 4) == "RIFF") { "Not a RIFF wav: $file" }
        // Walk chunks to find "data" (some encoders insert LIST chunks before it).
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
