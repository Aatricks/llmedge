package io.aatricks.llmedge

import io.aatricks.llmedge.model.ModelRegistry
import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.text.runtime.SmolLM

data class LLMEdgeConfig(
    val models: ModelRegistry = ModelRegistry(),
    val preferPerformanceMode: Boolean = false,
    val textCacheSize: Int = 2,
    val textCacheMemoryMb: Long = 2048,
    val speechCacheSize: Int = 1,
    val speechCacheMemoryMb: Long = 1024,
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
)
