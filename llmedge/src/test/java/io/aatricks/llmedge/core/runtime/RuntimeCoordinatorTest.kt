package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.runtime.ModelCache
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeCoordinatorTest {
    private data class FakeOptions(
        val allowGpu: Boolean,
        val openClAvailable: Boolean = false,
        val vulkanAvailable: Boolean = false,
    )

    private class FakeRuntime(
        val backend: ComputeBackend,
        private val onClose: () -> Unit = {},
    ) : ManagedRuntime {
        override val mutex = Mutex()

        override fun estimatedSizeBytes(): Long = 1L

        override fun close() {
            onClose()
        }
    }

    @After
    fun tearDown() {
        BackendRuntimePolicy.resetForTests()
    }

    @Test
    fun `acquire reuses cached runtime for identical spec and options`() = runTest {
        var loadCalls = 0
        val coordinator = createCoordinator { _, _, backend ->
            loadCalls++
            FakeRuntime(backend)
        }

        val first = coordinator.acquire("model", FakeOptions(allowGpu = false))
        val second = coordinator.acquire("model", FakeOptions(allowGpu = false))

        assertSame(first, second)
        assertEquals(1, loadCalls)
    }

    @Test
    fun `backend failure invalidates blacklisted runtime and reloads on next candidate`() = runTest {
        var loadCalls = 0
        var closeCalls = 0
        val coordinator = createCoordinator { _, _, backend ->
            loadCalls++
            FakeRuntime(backend) { closeCalls++ }
        }

        val options = FakeOptions(allowGpu = true, openClAvailable = true)
        val first = coordinator.acquire("model", options)
        assertEquals(ComputeBackend.OPENCL, first.backend)

        val blacklisted =
            coordinator.recordBackendFailureIfNeeded(
                spec = "model",
                options = options,
                runtime = first,
                error = IllegalStateException("backend device lost"),
            )
        val second = coordinator.acquire("model", options)

        assertTrue(blacklisted)
        assertEquals(ComputeBackend.CPU, second.backend)
        assertEquals(2, loadCalls)
        assertEquals(1, closeCalls)
    }

    @Test
    fun `pooled runtime loading tries opencl then vulkan before cpu`() = runTest {
        val attemptedBackends = mutableListOf<ComputeBackend>()
        val coordinator = createCoordinator { _, _, backend ->
            attemptedBackends += backend
            if (backend == ComputeBackend.OPENCL) {
                throw IllegalStateException("opencl init failed")
            }
            FakeRuntime(backend)
        }

        val runtime =
            coordinator.acquire(
                "model",
                FakeOptions(allowGpu = true, openClAvailable = true, vulkanAvailable = true),
            )

        assertEquals(listOf(ComputeBackend.OPENCL, ComputeBackend.VULKAN), attemptedBackends)
        assertEquals(ComputeBackend.VULKAN, runtime.backend)
    }

    private fun createCoordinator(
        loader: suspend (String, FakeOptions, ComputeBackend) -> FakeRuntime,
    ): RuntimeCoordinator<String, FakeOptions, FakeRuntime> =
        RuntimeCoordinator(
            cache = ModelCache(maxCacheSize = 2, maxMemoryMB = 16),
            cacheKeyPrefix = { spec, options ->
                RuntimeCacheKeyBuilder.prefix(spec, options.allowGpu, options.openClAvailable, options.vulkanAvailable)
            },
            loadRuntime = loader,
            activeBackend = { it.backend },
            candidateRequest = { options ->
                BackendCandidateResolver.Request(
                    subsystem = ComputeSubsystem.TEXT,
                    allowGpu = options.allowGpu,
                    openClAvailable = options.openClAvailable,
                    vulkanAvailable = options.vulkanAvailable,
                )
            },
        )
}
