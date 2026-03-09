package io.aatricks.llmedge.benchmark

import android.os.Debug
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.SmolLM
import org.junit.After
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Benchmarks for peak native memory usage during model operations.
 *
 * Requires a model file on device — set the model path via instrumentation args:
 *   -e llmedge.benchmark.model_path /data/local/tmp/model.gguf
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class MemoryBenchmark {

    private var modelPath: String? = null

    @Before
    fun setUp() {
        modelPath =
            InstrumentationRegistry.getInstrumentation()
                .arguments
                .getString("llmedge.benchmark.model_path")
        BenchmarkReporter.clear()
    }

    @After
    fun tearDown() {
        BenchmarkReporter.printSummary()
    }

    @Test
    fun peakMemoryDuringLoad() {
        assumeTrue("No model path provided", modelPath != null)
        val path = modelPath!!

        val baselineHeap = Debug.getNativeHeapAllocatedSize()
        BenchmarkReporter.record(
            "memory", "baseline_native_heap", baselineHeap / (1024.0 * 1024.0), "MB"
        )

        val model = SmolLM(useVulkan = false)
        model.loadModel(path, contextSize = 2048)

        val afterLoad = Debug.getNativeHeapAllocatedSize()
        BenchmarkReporter.record(
            "memory", "after_load_native_heap", afterLoad / (1024.0 * 1024.0), "MB"
        )
        BenchmarkReporter.record(
            "memory", "model_memory_delta", (afterLoad - baselineHeap) / (1024.0 * 1024.0), "MB"
        )

        // Generate some tokens to measure peak during inference
        model.completionInit("Write a poem about the moon.")
        model.completionLoop(64)

        val peakDuringInference = Debug.getNativeHeapAllocatedSize()
        BenchmarkReporter.record(
            "memory", "peak_during_inference", peakDuringInference / (1024.0 * 1024.0), "MB"
        )
        BenchmarkReporter.record(
            "memory",
            "inference_memory_delta",
            (peakDuringInference - afterLoad) / (1024.0 * 1024.0),
            "MB"
        )

        model.close()

        val afterClose = Debug.getNativeHeapAllocatedSize()
        BenchmarkReporter.record(
            "memory", "after_close_native_heap", afterClose / (1024.0 * 1024.0), "MB"
        )
        BenchmarkReporter.record(
            "memory",
            "memory_freed_on_close",
            (peakDuringInference - afterClose) / (1024.0 * 1024.0),
            "MB"
        )
    }
}
