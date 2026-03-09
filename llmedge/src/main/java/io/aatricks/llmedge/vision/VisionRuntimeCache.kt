package io.aatricks.llmedge.vision

import io.aatricks.llmedge.SmolLM
import io.aatricks.llmedge.core.AndroidLogAdapter

/**
 * Cache for warmed vision runtimes (SmolLM + Projector pairs) to avoid
 * cold-starting the model on every vision request.
 *
 * Keyed by (modelPath, projectorPath). Holds at most [maxEntries] entries;
 * when full, evicts the least-recently-used entry.
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
            try { projector.close() } catch (e: Exception) {
                AndroidLogAdapter.w(TAG, "Error closing projector: ${e.message}")
            }
            try { smolLM.close() } catch (e: Exception) {
                AndroidLogAdapter.w(TAG, "Error closing SmolLM: ${e.message}")
            }
        }
    }

    private val cache = LinkedHashMap<CacheKey, CachedRuntime>(4, 0.75f, true)

    @Synchronized
    fun get(key: CacheKey): CachedRuntime? {
        val entry = cache[key]
        if (entry != null) {
            entry.lastUsedMs = System.currentTimeMillis()
            AndroidLogAdapter.d(TAG, "Cache HIT for ${key.modelPath}")
        }
        return entry
    }

    @Synchronized
    fun put(key: CacheKey, runtime: CachedRuntime) {
        while (cache.size >= maxEntries) {
            val lruKey = cache.keys.first()
            val evicted = cache.remove(lruKey)
            AndroidLogAdapter.i(TAG, "Evicting vision runtime for ${lruKey.modelPath}")
            evicted?.close()
        }
        cache[key] = runtime
        AndroidLogAdapter.i(TAG, "Cached vision runtime for ${key.modelPath}")
    }

    @Synchronized
    fun releaseAll() {
        AndroidLogAdapter.i(TAG, "Releasing all cached vision runtimes (${cache.size} entries)")
        cache.values.forEach { it.close() }
        cache.clear()
    }
}
