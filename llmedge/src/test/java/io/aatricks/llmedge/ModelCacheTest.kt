package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ModelCacheTest {
    private class DummyModel : AutoCloseable {
        var closed = false
            private set

        override fun close() {
            closed = true
        }
    }

    @Test
    fun `system memory budget can trigger eviction before cache size limit`() {
        val cache = ModelCache<DummyModel>(maxCacheSize = 4, maxMemoryMB = 1024)
        cache.systemMemoryProvider = { 300L }

        val first = DummyModel()
        val second = DummyModel()

        cache.put("first", first, 200L * 1024L * 1024L)
        cache.put("second", second, 200L * 1024L * 1024L)

        assertTrue(first.closed)
        assertFalse(second.closed)
        assertFalse(cache.contains("first"))
        assertTrue(cache.contains("second"))
        assertEquals(1, cache.getStats().evictions)
    }

    @Test
    fun `clear closes entries and resets stats`() {
        val cache = ModelCache<DummyModel>(maxCacheSize = 2, maxMemoryMB = 1024)
        val first = DummyModel()
        val second = DummyModel()

        cache.put("first", first, 10L)
        cache.put("second", second, 20L)
        cache.get("first")
        cache.get("missing")

        cache.clear()

        assertTrue(first.closed)
        assertTrue(second.closed)
        assertEquals(0, cache.getStats().entries)
        assertEquals(0, cache.getStats().hits)
        assertEquals(0, cache.getStats().misses)
        assertEquals(0, cache.getStats().evictions)
    }
}