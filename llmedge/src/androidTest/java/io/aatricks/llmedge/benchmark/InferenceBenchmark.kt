package io.aatricks.llmedge.benchmark

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Benchmarks for text inference latency and throughput.
 *
 * Requires a model file on device — set the model path via instrumentation args:
 *   -e llmedge.benchmark.model_path /data/local/tmp/model.gguf
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class InferenceBenchmark {

    private lateinit var context: Context
    private var modelPath: String? = null
    private var model: SmolLM? = null

    companion object {
        private const val WARMUP_PROMPT = "Hello"
        private const val BENCH_PROMPT = "Explain the theory of relativity in simple terms."
        private const val MAX_TOKENS = 128
    }

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        modelPath =
            InstrumentationRegistry.getArguments()
                .getString("llmedge.benchmark.model_path")
        BenchmarkReporter.clear()
    }

    @After
    fun tearDown() {
        BenchmarkReporter.printSummary()
        model?.close()
        model = null
    }

    @Test
    fun firstTokenLatency() {
        assumeTrue("No model path provided", modelPath != null)
        val path = modelPath!!

        val smol = SmolLM(useVulkan = false)
        runBlocking { smol.load(path, SmolLM.InferenceParams(contextSize = 2048)) }
        model = smol

        // Warmup
        smol.getResponse(WARMUP_PROMPT, maxTokens = 16)
        smol.clearKvCache()

        // Measure first-token latency and throughput in a single streaming pass
        var firstTokenMs = 0.0
        var tokenCount = 0
        val startNs = System.nanoTime()
        runBlocking {
            var isFirst = true
            smol.getResponseAsFlow(BENCH_PROMPT).take(MAX_TOKENS).collect {
                tokenCount++
                if (isFirst) {
                    firstTokenMs = (System.nanoTime() - startNs) / 1_000_000.0
                    isFirst = false
                }
            }
        }
        val totalMs = (System.nanoTime() - startNs) / 1_000_000.0

        BenchmarkReporter.record("inference", "first_token_latency", firstTokenMs, "ms")
        val throughputMs = (totalMs - firstTokenMs).coerceAtLeast(1.0)
        val tokensPerSec = (tokenCount - 1).coerceAtLeast(0) / (throughputMs / 1000.0)
        BenchmarkReporter.record("inference", "tokens_per_second", tokensPerSec, "tok/s")
        BenchmarkReporter.record("inference", "generation_time", throughputMs, "ms")
        BenchmarkReporter.recordNativeMemoryMB("inference", "peak_native_heap")
    }
}
