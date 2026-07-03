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
    val useVulkan: Boolean = true,
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
) {
    init {
        require(promptThreads > 0) { "Text prompt threads must be positive." }
        require(generationThreads > 0) { "Text generation threads must be positive." }
        require(batchSize > 0) { "Text batch size must be positive." }
        require(streamBatchSize > 0) { "Text stream batch size must be positive." }
    }
}

data class SpeechRuntimeConfig(
    val cache: RuntimeCacheConfig = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 1024),
)

data class ImageRuntimeConfig(
    val cache: RuntimeCacheConfig = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 4096),
    val preferPerformanceMode: Boolean = false,
)

data class VisionRuntimeConfig(
    val cache: RuntimeCacheConfig = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 2048),
    val useVulkan: Boolean = true,
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
