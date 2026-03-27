package io.aatricks.llmedge.vision

import io.mockk.mockk
import io.mockk.verify
import org.junit.After
import org.junit.Before
import org.junit.Test

class VisionRuntimeCacheTest {
    @Before
    fun setUp() {
        System.setProperty("llmedge.disableNativeLoad", "true")
    }

    @After
    fun tearDown() {
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `put evicts previous runtime when cache is single entry`() {
        val cache = VisionRuntimeCache()
        val first = VisionRuntimeCache.CachedRuntime(mockk(relaxed = true), mockk(relaxed = true))
        val second = VisionRuntimeCache.CachedRuntime(mockk(relaxed = true), mockk(relaxed = true))

        cache.put(VisionRuntimeCache.CacheKey("model-a", "proj-a"), first)
        cache.put(VisionRuntimeCache.CacheKey("model-b", "proj-b"), second)

        verify { first.projector.close() }
        verify { first.smolLM.close() }
    }
}
