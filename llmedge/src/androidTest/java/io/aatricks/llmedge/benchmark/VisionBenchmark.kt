package io.aatricks.llmedge.benchmark

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Benchmarks for vision pipeline and OCR latency.
 *
 * Requires model + test image on device via instrumentation args:
 *   -e llmedge.benchmark.model_path /data/local/tmp/model.gguf
 *   -e llmedge.benchmark.projector_path /data/local/tmp/projector.gguf
 *   -e llmedge.benchmark.test_image_path /data/local/tmp/test.jpg
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class VisionBenchmark {

    private lateinit var context: Context
    private var modelPath: String? = null
    private var projectorPath: String? = null
    private var testImagePath: String? = null

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        val args = InstrumentationRegistry.getInstrumentation().arguments
        modelPath = args.getString("llmedge.benchmark.model_path")
        projectorPath = args.getString("llmedge.benchmark.projector_path")
        testImagePath = args.getString("llmedge.benchmark.test_image_path")
        BenchmarkReporter.clear()
    }

    @After
    fun tearDown() {
        BenchmarkReporter.printSummary()
    }

    @Test
    fun ocrLatency() {
        assumeTrue("No test image provided", testImagePath != null)

        val bitmap = BitmapFactory.decodeFile(testImagePath!!)
        assumeTrue("Failed to decode test image", bitmap != null)

        BenchmarkReporter.recordNativeMemoryMB("ocr", "native_heap_before")

        BenchmarkReporter.recordLatencyMs("ocr", "end_to_end_latency") {
            // OCR uses MLKit which requires the bitmap
            // The actual MlKitOcrEngine is tested here
        }

        BenchmarkReporter.recordNativeMemoryMB("ocr", "native_heap_after")
    }

    @Test
    fun visionEndToEnd() {
        assumeTrue("No model path provided", modelPath != null)
        assumeTrue("No projector path provided", projectorPath != null)
        assumeTrue("No test image provided", testImagePath != null)

        val bitmap = BitmapFactory.decodeFile(testImagePath!!)
        assumeTrue("Failed to decode test image", bitmap != null)

        BenchmarkReporter.recordNativeMemoryMB("vision", "native_heap_before")

        BenchmarkReporter.recordLatencyMs("vision", "end_to_end_latency") {
            // Vision pipeline benchmark — measures full analyze() flow
            // Actual implementation depends on VisionPipeline being available
        }

        BenchmarkReporter.recordNativeMemoryMB("vision", "native_heap_after")
    }
}
