package io.aatricks.llmedge

import io.aatricks.llmedge.model.ModelRegistry
import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.text.runtime.SmolLM

data class RuntimeCacheConfig(
    val maxEntries: Int = 1,
    val maxMemoryMb: Long = 1024,
) {
    init {
        require(maxEntries > 0) { "Runtime cache must allow at least one entry." }
        require(maxMemoryMb > 0L) { "Runtime cache memory budget must be positive." }
    }
}

data class ExecutionConfig(
    val inferenceThreads: Int =
        CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
) {
    init {
        require(inferenceThreads > 0) { "Shared inference thread count must be positive." }
    }
}

data class TextRuntimeConfig(
    val cache: RuntimeCacheConfig = RuntimeCacheConfig(maxEntries = 2, maxMemoryMb = 2048),
    /**
     * CPU by default: the vendored ik_llama.cpp fork's strength is its CPU (IQK) kernels, and
     * its Vulkan path measured far slower on-device (S22 Xclipse 920: smollm-135M 4.4 vs 78
     * tok/s, Qwen3-0.6B 3.4 vs 22.6 tok/s). Opt in per-host on hardware where Vulkan wins.
     */
    val useVulkan: Boolean = false,
    val promptThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
    val generationThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.TOKEN_GENERATION),
    val batchSize: Int = SmolLM.DEFAULT_BLOCKING_BATCH_SIZE,
    val streamBatchSize: Int = 1,
    val contextSize: Long? = null,
    val minP: Float = 0.1f,
    val temperature: Float = 0.8f,
    val useMmap: Boolean = true,
    val useMlock: Boolean = false,
    val useFlashAttention: Boolean = true,
    /**
     * KV cache type for keys. [SmolLM.KvCacheType.Q8_KV] measured +4-6% decode and ~-12% load
     * heap vs F16 on S22 CPU, but is CPU-only (GPU load attempts fall back to CPU), so the
     * SDK default stays F16.
     */
    val kvCacheTypeK: SmolLM.KvCacheType = SmolLM.KvCacheType.DEFAULT,
    /** KV cache type for values. Q8_KV is rejected here (K-cache-only layout). */
    val kvCacheTypeV: SmolLM.KvCacheType = SmolLM.KvCacheType.DEFAULT,
    /**
     * Micro-batch size for prompt processing (llama.cpp n_ubatch). 0 = engine default (128).
     * Larger values trade compute-buffer memory for prefill speed on long (e.g. RAG) prompts:
     * S22 measured 27.5s -> 14.5s TTFT on a ~1800-token prompt going 128 -> 512.
     */
    val nUbatch: Int = CpuTopology.recommendBatchSize(CpuTopology.TaskType.PROMPT_PROCESSING),
) {
    init {
        require(promptThreads > 0) { "Text prompt threads must be positive." }
        require(generationThreads > 0) { "Text generation threads must be positive." }
        require(batchSize > 0) { "Text batch size must be positive." }
        require(streamBatchSize > 0) { "Text stream batch size must be positive." }
        require(nUbatch >= 0) { "Text nUbatch must be >= 0." }
    }
}

data class SpeechRuntimeConfig(
    val cache: RuntimeCacheConfig = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 1024),
)

data class ImageRuntimeConfig(
    val cache: RuntimeCacheConfig = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 4096),
    val preferPerformanceMode: Boolean = false,
    /**
     * GPU (Vulkan, or OpenCL where built in) stays eligible by default: diffusion is compute-bound
     * and the sd.cpp Vulkan path is the fast one on most devices — unlike text/vision, whose ik
     * fork favors CPU. Set false to force the CPU backend for image/video generation: the escape
     * hatch for devices whose Vulkan driver loads fine but deadlocks at the first compute dispatch
     * (observed on PowerVR DXT-48, Pixel 10 / Tensor G5), where no automatic fallback can trigger.
     */
    val useVulkan: Boolean = true,
)

data class VisionRuntimeConfig(
    val cache: RuntimeCacheConfig = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 2048),
    /** CPU by default for the same reason as [TextRuntimeConfig.useVulkan] (same engine). */
    val useVulkan: Boolean = false,
    val promptThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
    val generationThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.TOKEN_GENERATION),
    val useFlashAttention: Boolean = true,
) {
    init {
        require(promptThreads > 0) { "Vision prompt threads must be positive." }
        require(generationThreads > 0) { "Vision generation threads must be positive." }
    }
}

data class LLMEdgeConfig(
    val execution: ExecutionConfig = ExecutionConfig(),
    val models: ModelRegistry = ModelRegistry(),
    val text: TextRuntimeConfig = TextRuntimeConfig(),
    val speech: SpeechRuntimeConfig = SpeechRuntimeConfig(),
    val image: ImageRuntimeConfig = ImageRuntimeConfig(),
    val vision: VisionRuntimeConfig = VisionRuntimeConfig(),
)
