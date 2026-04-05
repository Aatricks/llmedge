package io.aatricks.llmedge.core.runtime

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.RuntimeCacheConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.runtime.ComputeBackend
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class RuntimePoolFactoryTest {
    private class FakeRuntime : ManagedRuntime {
        override val mutex = Mutex()
        var closed = false

        override fun estimatedSizeBytes(): Long = 1L

        override fun close() {
            closed = true
        }
    }

    @Test
    fun `createCachedRuntimePool reuses runtimes for matching keys`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val loads = mutableListOf<Pair<String, Int>>()
        val edgeScope = LLMEdgeScope(this, 1)

        try {
            val pool =
                createCachedRuntimePool(
                    context = context,
                    scope = edgeScope,
                    profile =
                        runtimePoolProfile(
                            cacheConfig = RuntimeCacheConfig(maxEntries = 2, maxMemoryMb = 256),
                            cacheKeyPrefix = { spec: String, options: Int -> "$spec:$options" },
                            loadRuntime = { spec, options, _ ->
                                loads += spec to options
                                FakeRuntime()
                            },
                            activeBackend = { ComputeBackend.CPU },
                            candidateRequest = {
                                BackendCandidateResolver.Request(
                                    subsystem = null,
                                    allowGpu = false,
                                    openClAvailable = false,
                                    vulkanAvailable = false,
                                )
                            },
                        ),
                )

            val first = pool.coordinator.acquire("model", 1)
            val reused = pool.coordinator.acquire("model", 1)
            val second = pool.coordinator.acquire("model", 2)

            assertSame(first, reused)
            assertNotSame(first, second)
            assertEquals(listOf("model" to 1, "model" to 2), loads)

            pool.close()
            advanceUntilIdle()
            assertTrue(first.closed)
            assertTrue(second.closed)
        } finally {
            edgeScope.close()
        }
    }
}
