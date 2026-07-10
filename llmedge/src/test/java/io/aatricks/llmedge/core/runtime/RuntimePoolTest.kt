package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.runtime.ModelCache
import java.lang.Thread.sleep
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimePoolTest {
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
    fun `acquire reuses cached runtime`() = runTest {
        var loadCalls = 0
        val pool = createPool { _, _, backend ->
            loadCalls++
            FakeRuntime(backend)
        }

        val first = pool.coordinator.acquire("model", FakeOptions(allowGpu = false))
        val second = pool.coordinator.acquire("model", FakeOptions(allowGpu = false))

        assertSame(first, second)
        assertEquals(1, loadCalls)
    }

    @Test
    fun `loadDetached bypasses cache`() = runTest {
        var loadCalls = 0
        val pool = createPool { _, _, backend ->
            loadCalls++
            FakeRuntime(backend)
        }

        val cached = pool.coordinator.acquire("model", FakeOptions(allowGpu = false))
        val detached = pool.coordinator.loadDetached("model", FakeOptions(allowGpu = false))

        assertNotSame(cached, detached)
        assertEquals(2, loadCalls)
    }

    @Test
    fun `acquireDetailed reports cold then warm runtime`() = runTest {
        val pool =
            createPool { _, _, backend ->
                sleep(5)
                FakeRuntime(backend)
            }

        val cold = pool.coordinator.acquireDetailed("model", FakeOptions(allowGpu = false))
        val warm = pool.coordinator.acquireDetailed("model", FakeOptions(allowGpu = false))

        assertFalse(cold.cacheHit)
        assertTrue(cold.modelLoadTimeMs > 0L)
        assertTrue(cold.acquireTimeMs >= cold.modelLoadTimeMs)
        assertTrue(warm.cacheHit)
        assertEquals(0L, warm.modelLoadTimeMs)
        assertSame(cold.runtime, warm.runtime)
    }

    @Test
    fun `backend failure blacklists failed runtime and reloads next backend`() = runTest {
        var loadCalls = 0
        var closeCalls = 0
        val pool = createPool { _, _, backend ->
            loadCalls++
            FakeRuntime(backend) { closeCalls++ }
        }

        val options = FakeOptions(allowGpu = true, openClAvailable = true)
        val first = pool.coordinator.acquire("model", options)
        val blacklisted =
            pool.coordinator.recordBackendFailureIfNeeded(
                "model",
                options,
                first,
                IllegalStateException("device lost"),
            )
        val second = pool.coordinator.acquire("model", options)

        assertTrue(blacklisted)
        assertEquals(ComputeBackend.CPU, second.backend)
        assertEquals(2, loadCalls)
        assertEquals(1, closeCalls)
    }

    @Test
    fun `acquireLeased does not pin a replacement entry with same key when identities differ`() = runTest {
        val pool = createPool { _, _, backend ->
            FakeRuntime(backend)
        }

        val options = FakeOptions(allowGpu = false)
        val spec = "model"

        val result = pool.coordinator.acquireDetailed(spec, options)
        val key = RuntimeCacheKeyBuilder.withBackend(result.keyPrefix, result.backend)

        val replacementInstance = FakeRuntime(ComputeBackend.CPU)
        pool.cache.put(key, replacementInstance, 1L)

        // pin with the old runtime instance must fail because the cache now holds replacementInstance
        val pinSuccessOld = pool.cache.pin(key, result.runtime)
        assertFalse("Should not pin when instance differs", pinSuccessOld)

        // pin with matching replacementInstance must succeed
        val pinSuccessNew = pool.cache.pin(key, replacementInstance)
        assertTrue("Should pin when instance matches", pinSuccessNew)
    }

    private fun createPool(
        loader: suspend (String, FakeOptions, ComputeBackend) -> FakeRuntime,
    ): RuntimePool<String, FakeOptions, FakeRuntime> =
        RuntimePool(
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
