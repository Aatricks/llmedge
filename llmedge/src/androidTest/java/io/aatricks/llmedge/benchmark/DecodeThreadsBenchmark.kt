package io.aatricks.llmedge.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Measures decode throughput at different generationThreads values on a real
 * device, to validate (or refute) the TOKEN_GENERATION = P-cores/2 heuristic
 * in [CpuTopology.getOptimalThreadCount].
 *
 * Run with:
 *   ./gradlew :llmedge:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.benchmark.DecodeThreadsBenchmark \
 *     -Pandroid.testInstrumentationRunnerArguments.llmedge.benchmark.model_path=/data/local/tmp/model.gguf
 */
@RunWith(AndroidJUnit4::class)
class DecodeThreadsBenchmark {
    companion object {
        private const val WARMUP_PROMPT = "Hello"
        private const val BENCH_PROMPT = "Explain the theory of relativity in simple terms."
        private const val MAX_TOKENS = 96
        private const val ITERATIONS = 3
    }

    @Test
    fun decodeThroughputByThreadCount() {
        val modelPath =
            InstrumentationRegistry.getArguments().getString("llmedge.benchmark.model_path")
        assumeTrue("No model path provided", modelPath != null)

        val coreInfo = CpuTopology.detectCoreTopology()
        val pCores = coreInfo.performanceCores
        val defaultThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.TOKEN_GENERATION)
        val candidates =
            listOf(defaultThreads, (pCores * 3) / 4, pCores, pCores + 2)
                .filter { it >= 1 }
                .distinct()
                .sorted()
        println("[DecodeThreadsBenchmark] topology=$coreInfo default=$defaultThreads candidates=$candidates")

        for (threads in candidates) {
            val smol = SmolLM(useVulkan = false)
            try {
                runBlocking {
                    smol.load(
                        modelPath!!,
                        SmolLM.InferenceParams(contextSize = 2048, generationThreads = threads),
                    )
                }
                smol.getResponse(WARMUP_PROMPT, maxTokens = 16)

                val speeds = DoubleArray(ITERATIONS)
                for (i in 0 until ITERATIONS) {
                    smol.clearMessages()
                    smol.clearKvCache()
                    smol.getResponse(BENCH_PROMPT, maxTokens = MAX_TOKENS)
                    val metrics = smol.getLastGenerationMetrics()
                    speeds[i] = metrics.tokensPerSecond.toDouble()
                }
                speeds.sort()
                val median = speeds[ITERATIONS / 2]
                println(
                    "[DecodeThreadsBenchmark] threads=$threads median=${"%.2f".format(median)} tok/s " +
                        "runs=${speeds.joinToString { "%.2f".format(it) }}",
                )
                BenchmarkReporter.record("decode_threads", "tok_s_threads_$threads", median, "tok/s")
            } finally {
                smol.close()
            }
        }
    }
}
