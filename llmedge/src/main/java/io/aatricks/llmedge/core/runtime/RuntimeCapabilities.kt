package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.VulkanDeviceInfo
import io.aatricks.llmedge.image.diffusion.StableDiffusion

internal object RuntimeCapabilities {
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
