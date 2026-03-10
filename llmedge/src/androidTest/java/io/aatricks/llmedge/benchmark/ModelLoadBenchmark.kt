package io.aatricks.llmedge.benchmark

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Benchmarks for model load latency and text inference throughput.
 *
 * Requires a model file on device — set the model path via instrumentation args:
 *   -e llmedge.benchmark.model_path /data/local/tmp/model.gguf
 *
 * Skip with: -e notAnnotation androidx.test.filters.LargeTest
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class ModelLoadBenchmark {

    private lateinit var context: Context
    private var modelPath: String? = null

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
    }

    @Test
    fun coldLoadLatency() {
        assumeTrue("No model path provided", modelPath != null)
        val path = modelPath!!

        BenchmarkReporter.recordNativeMemoryMB("cold_load", "native_heap_before")

        var model: SmolLM? = null
        BenchmarkReporter.recordLatencyMs("cold_load", "load_time") {
            model = SmolLM(useVulkan = false)
            runBlocking { model!!.load(path, SmolLM.InferenceParams(contextSize = 2048)) }
        }

        BenchmarkReporter.recordNativeMemoryMB("cold_load", "native_heap_after")
        model?.close()
    }

    @Test
    fun warmLoadLatency() {
        assumeTrue("No model path provided", modelPath != null)
        val path = modelPath!!

        // First load to warm filesystem caches
        val warmup = SmolLM(useVulkan = false)
        runBlocking { warmup.load(path, SmolLM.InferenceParams(contextSize = 2048)) }
        warmup.close()

        var model: SmolLM? = null
        BenchmarkReporter.recordLatencyMs("warm_load", "load_time") {
            model = SmolLM(useVulkan = false)
            runBlocking { model!!.load(path, SmolLM.InferenceParams(contextSize = 2048)) }
        }

        BenchmarkReporter.recordNativeMemoryMB("warm_load", "native_heap_after")
        model?.close()
    }
}
