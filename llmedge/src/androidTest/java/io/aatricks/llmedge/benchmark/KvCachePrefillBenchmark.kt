package io.aatricks.llmedge.benchmark

import android.os.Debug
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device measurements for the two deferred perf defaults:
 *  - KV cache type (F16 vs Q8_0 vs Q8_KV, K-only and K+V): decode tok/s + native heap after load.
 *  - Prompt prefill n_ubatch (128 vs 256 vs 512): wall time to first token on a RAG-sized prompt.
 *
 * Run with:
 *   ./gradlew :llmedge:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.benchmark.KvCachePrefillBenchmark \
 *     -Pandroid.testInstrumentationRunnerArguments.llmedge.benchmark.model_path=/data/local/tmp/smollm135.gguf
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class KvCachePrefillBenchmark {
    companion object {
        private const val WARMUP_PROMPT = "Hello"
        private const val BENCH_PROMPT = "Explain the theory of relativity in simple terms."
        private const val MAX_TOKENS = 96
        private const val ITERATIONS = 3
        private const val CONTEXT_SIZE = 4096L
    }

    private fun modelPath(): String? =
        InstrumentationRegistry.getArguments().getString("llmedge.benchmark.model_path")

    @Test
    fun decodeThroughputByKvCacheType() {
        val path = modelPath()
        assumeTrue("No model path provided", path != null)

        data class Config(
            val name: String,
            val k: SmolLM.KvCacheType,
            val v: SmolLM.KvCacheType,
            val vulkan: Boolean = false,
        )
        // Q8_KV for the V cache is rejected in InferenceParams: it SIGABRTs natively.
        val configs =
            listOf(
                Config("f16_f16", SmolLM.KvCacheType.DEFAULT, SmolLM.KvCacheType.DEFAULT),
                Config("q8_0_k_only", SmolLM.KvCacheType.Q8_0, SmolLM.KvCacheType.DEFAULT),
                Config("q8_kv_k_only", SmolLM.KvCacheType.Q8_KV, SmolLM.KvCacheType.DEFAULT),
                Config("vk_f16_f16", SmolLM.KvCacheType.DEFAULT, SmolLM.KvCacheType.DEFAULT, vulkan = true),
                Config("vk_q8_kv_k_only", SmolLM.KvCacheType.Q8_KV, SmolLM.KvCacheType.DEFAULT, vulkan = true),
            )

        for (config in configs) {
            val smol = SmolLM(useVulkan = config.vulkan)
            try {
                val heapBefore = Debug.getNativeHeapAllocatedSize()
                runBlocking {
                    smol.load(
                        path!!,
                        SmolLM.InferenceParams(
                            contextSize = CONTEXT_SIZE,
                            kvCacheTypeK = config.k,
                            kvCacheTypeV = config.v,
                        ),
                    )
                }
                val heapAfter = Debug.getNativeHeapAllocatedSize()
                BenchmarkReporter.record(
                    "kv_cache_type",
                    "load_heap_delta_${config.name}",
                    (heapAfter - heapBefore) / (1024.0 * 1024.0),
                    "MB",
                )

                smol.getResponse(WARMUP_PROMPT, maxTokens = 16)
                val speeds = DoubleArray(ITERATIONS)
                for (i in 0 until ITERATIONS) {
                    smol.clearMessages()
                    smol.clearKvCache()
                    smol.getResponse(BENCH_PROMPT, maxTokens = MAX_TOKENS)
                    speeds[i] = smol.getLastGenerationMetrics().tokensPerSecond.toDouble()
                }
                speeds.sort()
                val median = speeds[ITERATIONS / 2]
                println(
                    "[KvCacheBenchmark] ${config.name} median=${"%.2f".format(median)} tok/s " +
                        "runs=${speeds.joinToString { "%.2f".format(it) }}",
                )
                BenchmarkReporter.record("kv_cache_type", "tok_s_${config.name}", median, "tok/s")
            } catch (t: Throwable) {
                println("[KvCacheBenchmark] ${config.name} FAILED: ${t.message}")
                BenchmarkReporter.record("kv_cache_type", "failed_${config.name}", 1.0, "flag")
            } finally {
                smol.close()
            }
        }
        BenchmarkReporter.printSummary()
    }

    @Test
    fun prefillLatencyByUbatch() {
        val path = modelPath()
        assumeTrue("No model path provided", path != null)

        // RAG-sized prompt: ~1300 tokens of plain English.
        val paragraph =
            "Retrieval augmented generation combines a vector store with a language model. " +
                "The store returns the most similar chunks for a query and the model answers " +
                "using only that context. Chunk size, overlap and embedding quality all affect " +
                "the final answer accuracy in practice on mobile devices. "
        val prompt = buildString {
            append("Summarize the following notes in one sentence.\n\n")
            repeat(24) { append(paragraph) }
        }

        for (ubatch in listOf(128, 256, 512)) {
            val smol = SmolLM(useVulkan = false)
            try {
                runBlocking {
                    smol.load(
                        path!!,
                        SmolLM.InferenceParams(contextSize = CONTEXT_SIZE, nUbatch = ubatch),
                    )
                }
                // Warm the mmap'd weights so the first config doesn't pay page-in cost alone.
                smol.getResponse(WARMUP_PROMPT, maxTokens = 4)

                val times = DoubleArray(ITERATIONS)
                for (i in 0 until ITERATIONS) {
                    smol.clearMessages()
                    smol.clearKvCache()
                    val start = System.nanoTime()
                    smol.getResponse(prompt, maxTokens = 1)
                    times[i] = (System.nanoTime() - start) / 1_000_000.0
                }
                times.sort()
                val median = times[ITERATIONS / 2]
                println(
                    "[PrefillBenchmark] ubatch=$ubatch median=${"%.0f".format(median)} ms " +
                        "runs=${times.joinToString { "%.0f".format(it) }}",
                )
                BenchmarkReporter.record("prefill_ubatch", "ttft_ms_ubatch_$ubatch", median, "ms")
            } catch (t: Throwable) {
                println("[PrefillBenchmark] ubatch=$ubatch FAILED: ${t.message}")
                BenchmarkReporter.record("prefill_ubatch", "failed_ubatch_$ubatch", 1.0, "flag")
            } finally {
                smol.close()
            }
        }
        BenchmarkReporter.printSummary()
    }
}
