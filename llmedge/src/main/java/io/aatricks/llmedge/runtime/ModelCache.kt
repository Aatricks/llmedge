package io.aatricks.llmedge.runtime

import io.aatricks.llmedge.core.AndroidLogAdapter
import java.util.LinkedHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch

/**
 * LRU cache for models with memory-aware eviction
 *
 * @param T Model type that implements AutoCloseable
 * @param maxCacheSize Maximum number of models to keep in cache
 * @param maxMemoryMB Maximum memory to use for cache (approximate)
 */
class ModelCache<T : AutoCloseable>(
    private val maxCacheSize: Int = 2,
    private val maxMemoryMB: Long = 4096,
    private val closeScope: CoroutineScope? = null,
    /** Optional provider to compute current available system memory (MB). If provided, cache will use this to be more memory-aware. */
    var systemMemoryProvider: (() -> Long)? = null
) {
    private val TAG = "ModelCache"

    /** Cache entry with metadata */
    data class CacheEntry<T>(
            val model: T,
            var sizeBytes: Long,
            val loadTimeMs: Long,
            val sizeProvider: (() -> Long)? = null,
            var lastUsedMs: Long = System.currentTimeMillis(),
            var hitCount: Int = 0
    )

    /** Cache statistics */
    data class CacheStats(
            val entries: Int,
            val totalSizeMB: Long,
            val hits: Int,
            val misses: Int,
            val evictions: Int
    ) {
        val hitRate: Double
            get() = if (hits + misses > 0) hits.toDouble() / (hits + misses) else 0.0

        override fun toString(): String {
            return "Cache: $entries entries, ${totalSizeMB}MB, " +
                    "hits=$hits, misses=$misses, evictions=$evictions, " +
                    "hit_rate=${String.format("%.1f%%", hitRate * 100)}"
        }
    }

    // LinkedHashMap with access-order for LRU
    private val cache = LinkedHashMap<String, CacheEntry<T>>(16, 0.75f, true)

    private val lock = ReentrantReadWriteLock()

    // Running total of cached bytes to avoid O(n) sumOf on every put/evict/query
    private var totalCachedBytes: Long = 0L

    // Statistics — atomic to allow reads without holding the write lock
    private val hits = AtomicInteger(0)
    private val misses = AtomicInteger(0)
    private val evictions = AtomicInteger(0)

    /**
     * Get model from cache.
     *
     * Access-order [LinkedHashMap] updates must happen under the write lock,
     * so cache reads intentionally take the write lock to keep LRU ordering and
     * entry-size refresh consistent.
     * @return model if found, null otherwise
     */
    fun get(key: String): T? {
        val entry = lock.write {
            val e = cache[key]
            if (e != null) {
                refreshEntrySize(e)
                e.lastUsedMs = System.currentTimeMillis()
                e.hitCount++
            }
            e
        }
        if (entry != null) {
            hits.incrementAndGet()
            AndroidLogAdapter.d(TAG, "Cache HIT for '$key' (used ${entry.hitCount} times)")
            return entry.model
        }
        misses.incrementAndGet()
        AndroidLogAdapter.d(TAG, "Cache MISS for '$key'")
        return null
    }

    /**
     * Put model into cache
     * @param key Cache key
     * @param model Model instance
     * @param sizeBytes Estimated model size in bytes
     * @param loadTimeMs Time taken to load the model
     */
    fun put(key: String, model: T, sizeBytes: Long, loadTimeMs: Long = 0) {
        put(key, model, sizeBytes, loadTimeMs, null)
    }

    fun put(
        key: String,
        model: T,
        sizeBytes: Long,
        loadTimeMs: Long = 0,
        sizeProvider: (() -> Long)? = null,
    ) {
        val toClose = mutableListOf<T>()
        lock.write {
            val resolvedSizeBytes = sizeProvider?.invoke()?.coerceAtLeast(0L) ?: sizeBytes
            // Always allow the first insert. Otherwise a large model can get stuck in an
            // impossible eviction loop when the cache is empty, and on Android this also keeps
            // runtime acquisition off the live-memory binder path for the cold-load case.
            if (cache.isEmpty()) {
                logOversizedInsertIfNeeded(key, resolvedSizeBytes)
            } else {
                while (cache.isNotEmpty() && shouldEvict(resolvedSizeBytes)) {
                    evictLRULocked(toClose)
                }
                if (cache.isEmpty() && shouldEvict(resolvedSizeBytes)) {
                    logOversizedInsertIfNeeded(key, resolvedSizeBytes)
                }
            }

            cache[key]?.let { oldEntry ->
                totalCachedBytes -= oldEntry.sizeBytes
                toClose.add(oldEntry.model)
            }

            val entry =
                    CacheEntry(
                            model = model,
                            sizeBytes = resolvedSizeBytes,
                            loadTimeMs = loadTimeMs,
                            sizeProvider = sizeProvider,
                            lastUsedMs = System.currentTimeMillis()
                    )

            cache[key] = entry
            totalCachedBytes += resolvedSizeBytes
            AndroidLogAdapter.i(TAG, "Cached '$key' (${resolvedSizeBytes / 1024 / 1024}MB, loaded in ${loadTimeMs}ms)")
            logStats()
        }
        closeModels(toClose)
    }

    /** Check if we should evict based on cache size and memory limits */
    private fun shouldEvict(newSizeBytes: Long): Boolean {
        refreshAllEntrySizes()
        return ModelCacheBudgetPolicy.shouldEvict(
            entryCount = cache.size,
            maxCacheSize = maxCacheSize,
            totalCachedBytes = totalCachedBytes,
            newSizeBytes = newSizeBytes,
            maxMemoryMB = maxMemoryMB,
            systemMemoryProvider = systemMemoryProvider,
        )
    }

    private fun logOversizedInsertIfNeeded(
        key: String,
        resolvedSizeBytes: Long,
    ) {
        if (ModelCacheBudgetPolicy.shouldLogOversizedInsert(resolvedSizeBytes, maxMemoryMB)) {
            AndroidLogAdapter.w(
                TAG,
                "Caching '$key' even though it exceeds the configured cache budget " +
                    "(${resolvedSizeBytes / 1024 / 1024}MB > $maxMemoryMB MB) because the cache is empty",
            )
        }
    }

    private fun refreshEntrySize(entry: CacheEntry<T>) {
        val provider = entry.sizeProvider ?: return
        val refreshedSize = provider().coerceAtLeast(0L)
        if (refreshedSize == entry.sizeBytes) {
            return
        }
        totalCachedBytes += refreshedSize - entry.sizeBytes
        entry.sizeBytes = refreshedSize
    }

    private fun refreshAllEntrySizes() {
        cache.values.forEach(::refreshEntrySize)
    }

    /** Evict least recently used entry, closing the evicted model outside the lock */
    fun evictLRU() {
        val toClose = mutableListOf<T>()
        lock.write {
            evictLRULocked(toClose)
        }
        closeModels(toClose)
    }

    /** Internal eviction that collects models to close — must be called while holding write lock */
    private fun evictLRULocked(toClose: MutableList<T>) {
        if (cache.isEmpty()) return

        val lruKey = cache.keys.first()
        val lruEntry = cache.remove(lruKey)

        lruEntry?.let { entry ->
            totalCachedBytes -= entry.sizeBytes
            evictions.incrementAndGet()
            AndroidLogAdapter.i(
                    TAG,
                    "Evicted LRU '$lruKey' (used ${entry.hitCount} times, " +
                            "${entry.sizeBytes / 1024 / 1024}MB)"
            )
            toClose.add(entry.model)
        }
    }

    /** Close models outside any lock to avoid blocking concurrent readers */
    private fun closeModels(models: List<T>) {
        for (model in models) {
            if (closeScope != null) {
                closeScope.launch {
                    try {
                        model.close()
                    } catch (e: Exception) {
                        AndroidLogAdapter.w(TAG, "Error closing evicted entry: ${e.message}")
                    }
                }
            } else {
                try {
                    model.close()
                } catch (e: Exception) {
                    AndroidLogAdapter.w(TAG, "Error closing evicted entry: ${e.message}")
                }
            }
        }
    }

    /** Clear all cached models */
    fun clear() {
        val toClose: List<T>
        lock.write {
            AndroidLogAdapter.i(TAG, "Clearing cache (${cache.size} entries)")
            toClose = cache.values.map { it.model }
            cache.clear()
            totalCachedBytes = 0L
            hits.set(0)
            misses.set(0)
            evictions.set(0)
        }
        for (model in toClose) {
            try {
                model.close()
            } catch (e: Exception) {
                AndroidLogAdapter.w(TAG, "Error closing cache entry: ${e.message}")
            }
        }
    }

    /** Remove specific entry from cache */
    fun remove(key: String): Boolean {
        val entry: CacheEntry<T>?
        lock.write {
            entry = cache.remove(key)
            if (entry != null) {
                totalCachedBytes -= entry!!.sizeBytes
            }
        }
        if (entry != null) {
            try {
                entry!!.model.close()
                AndroidLogAdapter.i(TAG, "Removed '$key' from cache")
                return true
            } catch (e: Exception) {
                AndroidLogAdapter.w(TAG, "Error closing removed entry: ${e.message}")
            }
        }
        return false
    }

    /** Get cache statistics — reads atomic counters without needing a write lock */
    fun getStats(): CacheStats = lock.read {
        val totalSize = totalCachedBytes / 1024 / 1024
        CacheStats(
                entries = cache.size,
                totalSizeMB = totalSize,
                hits = hits.get(),
                misses = misses.get(),
                evictions = evictions.get()
        )
    }

    /** Log current cache statistics */
    private fun logStats() {
        val stats = getStats()
        AndroidLogAdapter.d(TAG, stats.toString())
    }

    /** Get current cache size in MB */
    fun getCurrentSizeMB(): Long = lock.read {
        totalCachedBytes / 1024 / 1024
    }

    /** Check if key exists in cache */
    fun contains(key: String): Boolean = lock.read {
        cache.containsKey(key)
    }
}
