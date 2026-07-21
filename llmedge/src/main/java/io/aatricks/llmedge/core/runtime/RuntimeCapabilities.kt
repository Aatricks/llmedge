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

    /**
     * GPU availability derived purely from the isolated-worker probe — never loads a GPU
     * driver in the host process. Used for the SmolLM-based stacks (text/vision) whose own
     * Vulkan check runs ggml backend registration in-process; that path GGML_ABORTs the host
     * on Vulkan < 1.2 drivers (e.g. Adreno 619) and cannot be made crash-safe from Kotlin.
     */
    fun probeDerivedAvailability(context: android.content.Context): ComputeBackendAvailability {
        val probe =
            WorkerBackendProber.cachedOrNull()
                ?: WorkerBackendProber.persistedOrNull(context.applicationContext)
        return ComputeBackendAvailability(
            openClAvailable = probe?.openClAvailable ?: false,
            vulkanAvailable = probe?.vulkanAvailable ?: false,
        )
    }

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
