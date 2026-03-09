package io.aatricks.llmedge.benchmark

import android.os.Debug
import android.util.Log
import java.util.Locale

/**
 * Utility for recording and reporting standardized benchmark metrics.
 * Outputs CSV-compatible lines to logcat for automated collection.
 */
object BenchmarkReporter {
    private const val TAG = "LLMEdgeBenchmark"

    data class TimingResult(
        val testName: String,
        val metricName: String,
        val value: Double,
        val unit: String,
    )

    private val results = mutableListOf<TimingResult>()

    fun record(testName: String, metricName: String, value: Double, unit: String) {
        val result = TimingResult(testName, metricName, value, unit)
        results.add(result)
        Log.i(TAG, formatCsv(result))
    }

    fun recordLatencyMs(testName: String, metricName: String, block: () -> Unit): Double {
        val start = System.nanoTime()
        block()
        val elapsed = (System.nanoTime() - start) / 1_000_000.0
        record(testName, metricName, elapsed, "ms")
        return elapsed
    }

    fun recordNativeMemoryMB(testName: String, metricName: String): Long {
        val mb = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)
        record(testName, metricName, mb.toDouble(), "MB")
        return mb
    }

    fun printSummary() {
        Log.i(TAG, "--- BENCHMARK SUMMARY ---")
        Log.i(TAG, "test_name,metric_name,value,unit")
        for (result in results) {
            Log.i(TAG, formatCsv(result))
        }
        Log.i(TAG, "--- END SUMMARY ---")
    }

    fun clear() {
        results.clear()
    }

    private fun formatCsv(result: TimingResult): String =
        "${result.testName},${result.metricName},${String.format(Locale.US, "%.3f", result.value)},${result.unit}"
}
