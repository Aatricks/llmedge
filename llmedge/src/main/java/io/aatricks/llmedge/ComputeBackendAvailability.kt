package io.aatricks.llmedge

/**
 * Snapshot of GPU backend availability for a specific llmedge subsystem.
 *
 * OpenCL and Vulkan availability are subsystem-specific because text, speech, image/video, and
 * vision are backed by different native engines.
 */
data class ComputeBackendAvailability(
    val openClAvailable: Boolean,
    val vulkanAvailable: Boolean,
    val vulkanDeviceInfo: VulkanDeviceInfo? = null,
)
