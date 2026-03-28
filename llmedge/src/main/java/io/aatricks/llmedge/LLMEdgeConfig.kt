package io.aatricks.llmedge

import io.aatricks.llmedge.model.ModelRegistry
import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.text.runtime.SmolLM

data class RuntimeCacheConfig(
    val maxEntries: Int,
    val maxMemoryMb: Long,
)

data class TextRuntimeConfig(
    val cache: RuntimeCacheConfig,
    val useVulkan: Boolean,
    val promptThreads: Int,
    val generationThreads: Int,
    val batchSize: Int,
    val streamBatchSize: Int,
    val contextSize: Long?,
    val minP: Float,
    val temperature: Float,
    val useMmap: Boolean,
    val useMlock: Boolean,
    val useFlashAttention: Boolean,
)

data class SpeechRuntimeConfig(
    val cache: RuntimeCacheConfig,
)

data class ImageRuntimeConfig(
    val cache: RuntimeCacheConfig,
    val preferPerformanceMode: Boolean,
)

data class LLMEdgeConfig(
    val models: ModelRegistry = ModelRegistry(),
    val preferPerformanceMode: Boolean = false,
    val textCacheSize: Int = 2,
    val textCacheMemoryMb: Long = 2048,
    val speechCacheSize: Int = 1,
    val speechCacheMemoryMb: Long = 1024,
    val imageCacheSize: Int = 1,
    val imageCacheMemoryMb: Long = 4096,
    val textUseVulkan: Boolean = true,
    val defaultTextThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
    val defaultTextGenerationThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.TOKEN_GENERATION),
    val defaultTextBatchSize: Int = SmolLM.DEFAULT_BLOCKING_BATCH_SIZE,
    val defaultTextStreamBatchSize: Int = 4,
    val defaultTextContextSize: Long? = null,
    val defaultTextMinP: Float = 0.1f,
    val defaultTextTemperature: Float = 0.8f,
    val defaultUseMmap: Boolean = true,
    val defaultUseMlock: Boolean = false,
    val defaultUseFlashAttention: Boolean = true,
) {
    val text: TextRuntimeConfig =
        TextRuntimeConfig(
            cache = RuntimeCacheConfig(textCacheSize, textCacheMemoryMb),
            useVulkan = textUseVulkan,
            promptThreads = defaultTextThreads.coerceAtLeast(1),
            generationThreads = defaultTextGenerationThreads.coerceAtLeast(1),
            batchSize = defaultTextBatchSize.coerceAtLeast(1),
            streamBatchSize = defaultTextStreamBatchSize.coerceAtLeast(1),
            contextSize = defaultTextContextSize,
            minP = defaultTextMinP,
            temperature = defaultTextTemperature,
            useMmap = defaultUseMmap,
            useMlock = defaultUseMlock,
            useFlashAttention = defaultUseFlashAttention,
        )

    val speech: SpeechRuntimeConfig =
        SpeechRuntimeConfig(
            cache = RuntimeCacheConfig(speechCacheSize, speechCacheMemoryMb),
        )

    val image: ImageRuntimeConfig =
        ImageRuntimeConfig(
            cache = RuntimeCacheConfig(imageCacheSize, imageCacheMemoryMb),
            preferPerformanceMode = preferPerformanceMode,
        )
}
