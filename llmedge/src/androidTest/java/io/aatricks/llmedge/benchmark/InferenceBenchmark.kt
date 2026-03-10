package io.aatricks.llmedge.benchmark

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.text.runtime.SmolLM
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
            InstrumentationRegistry.getInstrumentation()
                .arguments
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
        smol.loadModel(path, contextSize = 2048)
        model = smol

        // Warmup
        smol.completionInit(WARMUP_PROMPT)
        smol.completionLoop(16)
        smol.kvCacheClear()

        // Measure first-token latency
        BenchmarkReporter.recordLatencyMs("inference", "first_token_latency") {
            smol.completionInit(BENCH_PROMPT)
            smol.completionLoop(1)
        }

        // Continue generating and measure throughput
        val startNs = System.nanoTime()
        val remaining = smol.completionLoop(MAX_TOKENS - 1)
        val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0
        val tokenCount = remaining.split(" ").size.coerceAtLeast(1)
        val tokensPerSec = if (elapsedMs > 0) tokenCount / (elapsedMs / 1000.0) else 0.0

        BenchmarkReporter.record("inference", "tokens_per_second", tokensPerSec, "tok/s")
        BenchmarkReporter.record("inference", "generation_time", elapsedMs, "ms")
        BenchmarkReporter.recordNativeMemoryMB("inference", "peak_native_heap")
    }
}
