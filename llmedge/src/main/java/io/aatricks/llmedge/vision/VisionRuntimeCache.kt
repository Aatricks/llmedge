package io.aatricks.llmedge.vision

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.runtime.SingleEntryRuntimeCache
import io.aatricks.llmedge.text.runtime.SmolLM

/**
 * Cache for warmed vision runtimes (SmolLM + Projector pairs).
 *
 * The current vision pipeline only reuses a single runtime at a time, so this cache intentionally
 * keeps one LRU-style entry and closes the previous runtime on replacement.
 */
internal class VisionRuntimeCache(private val maxEntries: Int = 1) {
    companion object {
        private const val TAG = "VisionRuntimeCache"
    }

    internal data class CacheKey(val modelPath: String, val projectorPath: String)

    internal class CachedRuntime(
        val smolLM: SmolLM,
        val projector: Projector,
        var lastUsedMs: Long = System.currentTimeMillis(),
    ) : AutoCloseable {
        override fun close() {
            try {
                projector.close()
            } catch (e: Exception) {
                AndroidLogAdapter.w(TAG, "Error closing projector: ${e.message}")
            }
            try {
                smolLM.close()
            } catch (e: Exception) {
                AndroidLogAdapter.w(TAG, "Error closing SmolLM: ${e.message}")
            }
        }
    }

    private val cache = SingleEntryRuntimeCache<CacheKey, CachedRuntime>(TAG)

    @Synchronized
    fun get(key: CacheKey): CachedRuntime? {
        val entry = cache.get(key)
        if (entry != null) {
            entry.lastUsedMs = System.currentTimeMillis()
            AndroidLogAdapter.d(TAG, "Cache HIT for ${key.modelPath}")
        }
        return entry
    }

    @Synchronized
    fun put(
        key: CacheKey,
        runtime: CachedRuntime,
    ) {
        check(maxEntries == 1) { "VisionRuntimeCache currently supports a single cached runtime." }
        cache.put(key, runtime)
        AndroidLogAdapter.i(TAG, "Cached vision runtime for ${key.modelPath}")
    }

    @Synchronized
    fun releaseAll() {
        AndroidLogAdapter.i(TAG, "Releasing all cached vision runtimes")
        cache.clear()
    }
}
