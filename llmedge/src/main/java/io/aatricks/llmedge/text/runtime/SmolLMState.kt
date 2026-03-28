package io.aatricks.llmedge.text.runtime

import io.aatricks.llmedge.runtime.ComputeBackend

internal class SmolLMState(
    useVulkanGpu: Boolean,
    defaultReasoningBudget: Int,
) {
    var nativePtr: Long = 0L
    var requestedLoadBackend: ComputeBackend? = null
    var selectedBackend: ComputeBackend =
        if (useVulkanGpu) {
            ComputeBackend.VULKAN
        } else {
            ComputeBackend.CPU
        }
    var currentThinkingMode: SmolLM.ThinkingMode = SmolLM.ThinkingMode.DEFAULT
    var currentReasoningBudget: Int = defaultReasoningBudget
    var loadedInferenceParams: SmolLM.InferenceParams? = null

    val useVulkanGpu: Boolean = useVulkanGpu

    fun reset(defaultReasoningBudget: Int) {
        nativePtr = 0L
        requestedLoadBackend = null
        selectedBackend =
            if (useVulkanGpu) {
                ComputeBackend.VULKAN
            } else {
                ComputeBackend.CPU
            }
        currentThinkingMode = SmolLM.ThinkingMode.DEFAULT
        currentReasoningBudget = defaultReasoningBudget
        loadedInferenceParams = null
    }
}
