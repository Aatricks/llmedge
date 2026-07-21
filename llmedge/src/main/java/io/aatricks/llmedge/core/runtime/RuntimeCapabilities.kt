package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.ComputeBackendAvailability
import io.aatricks.llmedge.VulkanDeviceInfo
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.ipc.WorkerBackendProber
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.text.runtime.SmolLM

internal object RuntimeCapabilities {
    fun textBackendAvailability(): ComputeBackendAvailability =
        ComputeBackendAvailability(
            openClAvailable = SmolLM.isOpenClAvailable(),
            vulkanAvailable = if (WorkerBackendProber.isVulkanQuarantined()) false else SmolLM.isVulkanBackendAvailable(),
        )

    fun speechBackendAvailability(): ComputeBackendAvailability =
        ComputeBackendAvailability(
            openClAvailable = Whisper.isOpenClAvailable(),
            vulkanAvailable = if (WorkerBackendProber.isVulkanQuarantined()) false else Whisper.isVulkanBackendAvailable(),
        )

    fun imageBackendAvailability(): ComputeBackendAvailability =
        WorkerBackendProber.cachedOrNull() ?: ComputeBackendAvailability(false, false, null)

    fun visionBackendAvailability(): ComputeBackendAvailability =
        ComputeBackendAvailability(
            openClAvailable = SmolLM.isOpenClAvailable(),
            vulkanAvailable = if (WorkerBackendProber.isVulkanQuarantined()) false else SmolLM.isVulkanBackendAvailable(),
        )

    fun isStableDiffusionVulkanAvailable(): Boolean {
        val deviceCount = StableDiffusion.getVulkanDeviceCount()
        if (deviceCount <= 0) {
            return false
        }
        val memory = StableDiffusion.getVulkanDeviceMemory(0) ?: return false
        return memory.size >= 2
    }

    fun isStableDiffusionOpenClAvailable(): Boolean = StableDiffusion.isOpenClAvailable()

    fun getStableDiffusionVulkanDeviceInfo(): VulkanDeviceInfo? {
        val deviceCount = StableDiffusion.getVulkanDeviceCount()
        if (deviceCount <= 0) {
            return null
        }
        val memory = StableDiffusion.getVulkanDeviceMemory(0) ?: return null
        if (memory.size < 2) {
            return null
        }
        return VulkanDeviceInfo(
            deviceCount = deviceCount,
            freeMemoryMB = memory[0] / (1024 * 1024),
            totalMemoryMB = memory[1] / (1024 * 1024),
            deviceIndex = 0,
        )
    }
}
