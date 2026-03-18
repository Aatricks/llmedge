package io.aatricks.llmedge.runtime

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Test

class BackendRuntimePolicyTest {
    @After
    fun tearDown() {
        BackendRuntimePolicy.resetForTests()
    }

    @Test
    fun `opencl is preferred over vulkan when both are available`() {
        val candidates =
            BackendRuntimePolicy.candidates(
                subsystem = ComputeSubsystem.TEXT,
                allowGpu = true,
                openClAvailable = true,
                vulkanAvailable = true,
            )

        assertEquals(
            listOf(ComputeBackend.OPENCL, ComputeBackend.VULKAN, ComputeBackend.CPU),
            candidates,
        )
    }

    @Test
    fun `blacklisted backend is skipped for that subsystem`() {
        BackendRuntimePolicy.blacklist(ComputeSubsystem.VIDEO, ComputeBackend.VULKAN)

        val videoCandidates =
            BackendRuntimePolicy.candidates(
                subsystem = ComputeSubsystem.VIDEO,
                allowGpu = true,
                openClAvailable = true,
                vulkanAvailable = true,
            )
        val textCandidates =
            BackendRuntimePolicy.candidates(
                subsystem = ComputeSubsystem.TEXT,
                allowGpu = true,
                openClAvailable = true,
                vulkanAvailable = true,
            )

        assertEquals(listOf(ComputeBackend.OPENCL, ComputeBackend.CPU), videoCandidates)
        assertEquals(
            listOf(ComputeBackend.OPENCL, ComputeBackend.VULKAN, ComputeBackend.CPU),
            textCandidates,
        )
    }

    @Test
    fun `cpu is the only candidate when gpu is disabled`() {
        val candidates =
            BackendRuntimePolicy.candidates(
                subsystem = ComputeSubsystem.WHISPER,
                allowGpu = false,
                openClAvailable = true,
                vulkanAvailable = true,
            )

        assertEquals(listOf(ComputeBackend.CPU), candidates)
    }
}
