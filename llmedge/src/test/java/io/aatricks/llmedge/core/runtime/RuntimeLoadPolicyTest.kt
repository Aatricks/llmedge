package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeLoadPolicyTest {
    @After
    fun tearDown() {
        BackendRuntimePolicy.resetForTests()
    }

    @Test
    fun `preferred backend falls back to cpu when requested`() {
        assertEquals(
            listOf(ComputeBackend.VULKAN, ComputeBackend.CPU),
            RuntimeLoadPolicy.candidates(ComputeBackend.VULKAN, includeCpuFallback = true),
        )
        assertEquals(
            listOf(ComputeBackend.OPENCL),
            RuntimeLoadPolicy.candidates(ComputeBackend.OPENCL, includeCpuFallback = false),
        )
    }

    @Test
    fun `record backend failure blacklists gpu backends only for shared requests`() {
        val request =
            BackendCandidateResolver.Request(
                subsystem = ComputeSubsystem.TEXT,
                allowGpu = true,
                openClAvailable = true,
                vulkanAvailable = true,
            )

        assertTrue(RuntimeLoadPolicy.recordBackendFailureIfNeeded(request, ComputeBackend.OPENCL))
        assertTrue(BackendRuntimePolicy.isBlacklisted(ComputeSubsystem.TEXT, ComputeBackend.OPENCL))
        assertFalse(RuntimeLoadPolicy.recordBackendFailureIfNeeded(request, ComputeBackend.CPU))
        assertFalse(
            RuntimeLoadPolicy.recordBackendFailureIfNeeded(
                request = request,
                backend = ComputeBackend.VULKAN,
                preferredBackend = ComputeBackend.VULKAN,
            ),
        )
    }
}
