package io.aatricks.llmedge.runtime

import io.aatricks.llmedge.core.AndroidLogAdapter
import java.util.LinkedHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
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
            var hitCount: Int = 0,
            var pinCount: Int = 0
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
                // Do NOT refresh the entry size here: sizeProvider makes native JNI
                // calls on the runtime handle, which may be mid-inference on another
                // thread (no native-side lock). Sizes are resolved at insert time.
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
                    // Pinned entries are skipped; stop when nothing evictable remains
                    // (the insert then intentionally overshoots the budget).
                    if (!evictLRULocked(toClose)) break
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
        // Entry sizes are intentionally not refreshed here: sizeProvider calls into
        // native code on runtimes that other coroutines may be actively using.
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

    /**
     * Pin an entry so LRU eviction skips it while a long-lived session (e.g. streaming
     * transcription) holds the runtime. Pins are advisory against eviction only:
     * [clear] and [remove] still close pinned entries.
     *
     * @return true if the entry was present and is now pinned
     */
    fun pin(key: String): Boolean = lock.write {
        val entry = cache[key] ?: return@write false
        entry.pinCount++
        true
    }

    /** Release one pin taken with [pin]. Safe to call for absent keys. */
    fun unpin(key: String) {
        lock.write {
            cache[key]?.let { entry ->
                if (entry.pinCount > 0) entry.pinCount--
            }
        }
    }

    /** Evict least recently used entry, closing the evicted model outside the lock */
    fun evictLRU() {
        val toClose = mutableListOf<T>()
        lock.write {
            evictLRULocked(toClose)
        }
        closeModels(toClose)
    }

    /**
     * Evict the least recently used unpinned entry — must be called while holding write lock.
     * @return true if an entry was evicted, false if every entry is pinned (or cache is empty)
     */
    private fun evictLRULocked(toClose: MutableList<T>): Boolean {
        val lruKey = cache.entries.firstOrNull { it.value.pinCount == 0 }?.key ?: return false
        val lruEntry = cache.remove(lruKey) ?: return false

        totalCachedBytes -= lruEntry.sizeBytes
        evictions.incrementAndGet()
        AndroidLogAdapter.i(
                TAG,
                "Evicted LRU '$lruKey' (used ${lruEntry.hitCount} times, " +
                        "${lruEntry.sizeBytes / 1024 / 1024}MB)"
        )
        toClose.add(lruEntry.model)
        return true
    }

    /** Close models outside any lock to avoid blocking concurrent readers */
    private fun closeModels(models: List<T>) {
        for (model in models) {
            if (closeScope != null) {
                // close() blocks until any in-flight generation on the runtime finishes;
                // force Dispatchers.IO so cache pressure can never park that wait on the
                // caller's dispatcher (often Main) — an ANR under normal usage otherwise.
                closeScope.launch(Dispatchers.IO) {
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

    /**
     * Clear all cached models. Closes synchronously — deferring the closes to a
     * scope would let a shutdown cancel them and leak native runtimes — and close()
     * waits for in-flight generations, so call this off the main thread.
     */
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

    /**
     * Remove specific entry from cache. Closes synchronously so callers (e.g. the
     * FLUX.2 sequential invalidate) can rely on the runtime being freed on return;
     * close() waits for in-flight generations, so call off the main thread.
     */
    fun remove(key: String): Boolean {
        val entry: CacheEntry<T>?
        lock.write {
            entry = cache.remove(key)
            if (entry != null) {
                totalCachedBytes -= entry.sizeBytes
            }
        }
        if (entry != null) {
            try {
                entry.model.close()
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
