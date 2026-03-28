package io.aatricks.llmedge.text.runtime.internal

import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.runtime.SmolLM

internal object SmolLMRuntimeConfigSupport {
    fun applyReasoningState(
        instance: SmolLM,
        mode: SmolLM.ThinkingMode,
        budget: Int,
    ) {
        val effectiveMode = if (budget == 0) SmolLM.ThinkingMode.DISABLED else mode
        instance.state.currentThinkingMode = effectiveMode
        instance.state.currentReasoningBudget = budget
        val nativePtr = instance.state.nativePtr
        if (nativePtr != 0L) {
            instance.bridge.setReasoningOptions(
                instance,
                nativePtr,
                effectiveMode.disableReasoning || budget == 0,
                budget,
            )
        }
    }

    fun resolvedReasoningBudget(
        mode: SmolLM.ThinkingMode,
        override: Int?,
        defaultReasoningBudget: Int,
    ): Int = override ?: if (mode.disableReasoning) 0 else defaultReasoningBudget

    fun resolveContextSize(
        requested: Long?,
        modelContextSize: Long,
        minContextSize: Long,
        defaultContextSizeCap: Long,
        onClamped: (desired: Long, clamped: Long, heapMb: Long) -> Unit,
    ): Long {
        if (requested != null) {
            return requested.coerceIn(minContextSize, defaultContextSizeCap)
        }
        val desired = modelContextSize
        val heapAwareCap = recommendedContextCap(defaultContextSizeCap)
        val effectiveCap = minOf(defaultContextSizeCap, heapAwareCap)
        val clamped = desired.coerceIn(minContextSize, effectiveCap)
        if (desired != clamped) {
            val heapMb = Runtime.getRuntime().maxMemory() / (1024 * 1024)
            onClamped(desired, clamped, heapMb)
        }
        return clamped
    }

    fun resolveChatTemplate(
        explicit: String?,
        ggufReader: GGUFReader,
        defaultTemplate: String,
    ): String = explicit ?: (ggufReader.getChatTemplate() ?: defaultTemplate)

    fun preflightBackendCompatibility(
        useVulkanGpu: Boolean,
        hasVulkanBackendSupport: Boolean,
        modelPath: String,
        params: SmolLM.InferenceParams,
        fileType: Int?,
        dominantTensorType: Int?,
        ggufFileTypeNames: Map<Int, String>,
        ggufTensorTypeNames: Map<Int, String>,
    ) {
        if (!useVulkanGpu || hasVulkanBackendSupport) {
            return
        }
        val detail =
            buildString {
                append("SmolLM was configured with useVulkan=true, but the active native build does not include Vulkan support.")
                append(" nGpuLayers=")
                append(params.nGpuLayers)
                append(", kvCacheTypeK=")
                append(params.kvCacheTypeK.name)
                append(", kvCacheTypeV=")
                append(params.kvCacheTypeV.name)
                append(", ggufFileType=")
                append(describeQuantizedValue(fileType, ggufFileTypeNames))
                append(", dominantTensorType=")
                append(describeQuantizedValue(dominantTensorType, ggufTensorTypeNames))
                append(". Disable Vulkan for CPU-only loading or install a Vulkan-enabled llmedge build.")
            }
        throw ModelLoadException(modelPath, detail)
    }

    private fun describeQuantizedValue(
        value: Int?,
        names: Map<Int, String>,
    ): String =
        when (value) {
            null -> "unknown"
            else -> names[value]?.let { "$it ($value)" } ?: "unknown($value)"
        }

    private fun recommendedContextCap(defaultContextSizeCap: Long): Long {
        val heapMb = Runtime.getRuntime().maxMemory() / (1024 * 1024)
        return when {
            heapMb <= 256 -> 2_048L
            heapMb <= 384 -> 4_096L
            heapMb <= 512 -> 6_144L
            else -> defaultContextSizeCap
        }
    }
}
