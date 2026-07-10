package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import io.aatricks.llmedge.runtime.ModelCache
import java.util.concurrent.atomic.AtomicBoolean

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

    @Test
    fun `entry sizes are fixed at insert and live runtimes are not re-queried`() {
        val cache = ModelCache<DummyModel>(maxCacheSize = 4, maxMemoryMB = 1024)
        cache.systemMemoryProvider = { 4096L }

        val first = DummyModel()
        val second = DummyModel()
        var providerCalls = 0
        var firstSizeBytes = 150L * 1024L * 1024L

        cache.put(
            key = "first",
            model = first,
            sizeBytes = firstSizeBytes,
            sizeProvider = {
                providerCalls++
                firstSizeBytes
            },
        )
        assertEquals(1, providerCalls)

        // The provider's answer growing later must NOT be observed by cache
        // operations: re-querying would make native JNI calls on a runtime that
        // may be mid-inference on another thread.
        firstSizeBytes = 300L * 1024L * 1024L
        cache.put(
            key = "second",
            model = second,
            sizeBytes = 100L * 1024L * 1024L,
        )
        cache.get("first")

        assertEquals(1, providerCalls)
        assertFalse(first.closed)
        assertFalse(second.closed)
        assertTrue(cache.contains("first"))
        assertTrue(cache.contains("second"))
        assertEquals(250L, cache.getCurrentSizeMB())
        assertEquals(0, cache.getStats().evictions)
    }

    @Test
    fun `first insert bypasses live system memory gate`() {
        val cache = ModelCache<DummyModel>(maxCacheSize = 1, maxMemoryMB = 4096)
        cache.systemMemoryProvider = { 64L }

        val first = DummyModel()

        cache.put("first", first, 1969L * 1024L * 1024L)

        assertFalse(first.closed)
        assertTrue(cache.contains("first"))
        assertEquals(1969L, cache.getCurrentSizeMB())
        assertEquals(0, cache.getStats().evictions)
    }

    @Test
    fun `pinned entry survives eviction pressure and unpin re-enables eviction`() {
        val cache = ModelCache<DummyModel>(maxCacheSize = 1, maxMemoryMB = 1024)

        val first = DummyModel()
        val second = DummyModel()
        val third = DummyModel()

        cache.put("first", first, 10L)
        assertTrue(cache.pin("first"))

        // maxCacheSize=1 would evict "first", but the pin protects it.
        cache.put("second", second, 10L)
        assertFalse(first.closed)
        assertTrue(cache.contains("first"))
        assertTrue(cache.contains("second"))

        cache.unpin("first")
        cache.put("third", third, 10L)
        // With the pin gone, LRU eviction drains back down.
        assertTrue(first.closed)
        assertTrue(cache.contains("third"))
    }

    @Test
    fun `pin returns false for absent keys and clear closes pinned entries`() {
        val cache = ModelCache<DummyModel>(maxCacheSize = 2, maxMemoryMB = 1024)
        assertFalse(cache.pin("missing"))
        cache.unpin("missing") // must not throw

        val first = DummyModel()
        cache.put("first", first, 10L)
        assertTrue(cache.pin("first"))

        cache.clear()
        assertTrue(first.closed)
        assertFalse(cache.contains("first"))
    }

    @Test
    fun `conditional pin matches exact instance`() {
        val cache = ModelCache<DummyModel>(maxCacheSize = 2, maxMemoryMB = 1024)
        val first = DummyModel()
        val replacement = DummyModel()

        cache.put("first", first, 10L)

        // pin with replacement model should fail
        assertFalse(cache.pin("first", replacement))

        // pin with correct model should succeed
        assertTrue(cache.pin("first", first))

        // Let's test that replacement cannot be pinned even if we put it afterward without unpinning
        cache.unpin("first")
        cache.put("first", replacement, 10L)
        assertFalse(cache.pin("first", first))
        assertTrue(cache.pin("first", replacement))
    }

    @Test
    fun `oversized insert after eviction does not loop forever`() {
        val cache = ModelCache<DummyModel>(maxCacheSize = 1, maxMemoryMB = 4096)
        cache.systemMemoryProvider = { 64L }

        val first = DummyModel()
        val second = DummyModel()
        cache.put("first", first, 32L * 1024L * 1024L)

        val completed = AtomicBoolean(false)
        val worker =
            Thread {
                cache.put("second", second, 1969L * 1024L * 1024L)
                completed.set(true)
            }.apply {
                isDaemon = true
                start()
            }

        worker.join(1000)

        assertTrue("Expected oversized insert to finish after draining the cache", completed.get())
        assertTrue(first.closed)
        assertFalse(second.closed)
        assertFalse(cache.contains("first"))
        assertTrue(cache.contains("second"))
    }
}
