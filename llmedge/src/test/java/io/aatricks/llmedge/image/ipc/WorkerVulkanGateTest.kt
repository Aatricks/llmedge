package io.aatricks.llmedge.image.ipc

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WorkerVulkanGateTest {
    @Test
    fun `vulkan stays enabled with no verdicts`() {
        assertFalse(WorkerVulkanGate.shouldDisable(useVulkan = true, blacklistSeed = emptyList()))
    }

    @Test
    fun `config disable wins`() {
        assertTrue(WorkerVulkanGate.shouldDisable(useVulkan = false, blacklistSeed = emptyList()))
    }

    @Test
    fun `a vulkan verdict disables the driver for the whole worker`() {
        assertTrue(
            WorkerVulkanGate.shouldDisable(
                useVulkan = true,
                blacklistSeed = listOf("IMAGE:VULKAN"),
            ),
        )
    }

    @Test
    fun `non-vulkan verdicts do not disable vulkan`() {
        assertFalse(
            WorkerVulkanGate.shouldDisable(
                useVulkan = true,
                blacklistSeed = listOf("IMAGE:OPENCL"),
            ),
        )
    }
}
