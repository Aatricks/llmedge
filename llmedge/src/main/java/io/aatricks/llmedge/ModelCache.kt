package io.aatricks.llmedge

import android.util.Log
import java.util.LinkedHashMap

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
    /** Optional provider to compute current available system memory (MB). If provided, cache will use this to be more memory-aware. */
    var systemMemoryProvider: (() -> Long)? = null
) {
    private val TAG = "ModelCache"

    /** Cache entry with metadata */
    data class CacheEntry<T>(
            val model: T,
            val sizeBytes: Long,
            val loadTimeMs: Long,
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

    // Running total of cached bytes to avoid O(n) sumOf on every put/evict/query
    private var totalCachedBytes: Long = 0L

    // Statistics
    private var hits = 0
    private var misses = 0
    private var evictions = 0

    /**
     * Get model from cache
     * @return model if found, null otherwise
     */
    @Synchronized
    fun get(key: String): T? {
        val entry = cache[key]
        if (entry != null) {
            entry.lastUsedMs = System.currentTimeMillis()
            entry.hitCount++
            hits++
            Log.d(TAG, "Cache HIT for '$key' (used ${entry.hitCount} times)")
            return entry.model
        }
        misses++
        Log.d(TAG, "Cache MISS for '$key'")
        return null
    }

    /**
     * Put model into cache
     * @param key Cache key
     * @param model Model instance
     * @param sizeBytes Estimated model size in bytes
     * @param loadTimeMs Time taken to load the model
     */
    @Synchronized
    fun put(key: String, model: T, sizeBytes: Long, loadTimeMs: Long = 0) {
        // Check if we need to evict
        while (shouldEvict(sizeBytes)) {
            evictLRU()
        }

        // Remove existing entry if present
        cache[key]?.let { oldEntry ->
            totalCachedBytes -= oldEntry.sizeBytes
            try {
                oldEntry.model.close()
            } catch (e: Exception) {
                Log.w(TAG, "Error closing old cache entry: ${e.message}")
            }
        }

        val entry =
                CacheEntry(
                        model = model,
                        sizeBytes = sizeBytes,
                        loadTimeMs = loadTimeMs,
                        lastUsedMs = System.currentTimeMillis()
                )

        cache[key] = entry
        totalCachedBytes += sizeBytes
        Log.i(TAG, "Cached '$key' (${sizeBytes / 1024 / 1024}MB, loaded in ${loadTimeMs}ms)")
        logStats()
    }

    /** Check if we should evict based on cache size and memory limits */
    private fun shouldEvict(newSizeBytes: Long): Boolean {
        if (cache.size >= maxCacheSize) return true

        val currentMemoryMB = totalCachedBytes / 1024 / 1024
        val newMemoryMB = currentMemoryMB + (newSizeBytes / 1024 / 1024)

        // If we have a system memory provider, be conservative and cap cache size to a fraction
        // of currently available system memory (e.g., reserve 10% of whatever is free)
        val effectiveMax = systemMemoryProvider?.let { provider ->
            val avail = provider()
            // Ensure we keep at least 10% of the available system memory for OS/other apps
            val reserved = (avail * 0.1).toLong()
            val budget = (avail - reserved).coerceAtMost(maxMemoryMB)
            // Avoid tiny budgets that cause immediate eviction; keep at least a conservative
            // lower bound so the cache can still hold large models when needed.
            val minBudget = (maxMemoryMB / 4).coerceAtLeast(256L)
            val finalBudget = budget.coerceAtLeast(minBudget)
            finalBudget
        } ?: maxMemoryMB

        return newMemoryMB > effectiveMax
    }

    /** Evict least recently used entry */
    @Synchronized
    fun evictLRU() {
        if (cache.isEmpty()) return

        // LinkedHashMap in access-order: first entry is LRU
        val lruKey = cache.keys.first()
        val lruEntry = cache.remove(lruKey)

        lruEntry?.let { entry ->
            totalCachedBytes -= entry.sizeBytes
            try {
                entry.model.close()
                evictions++
                Log.i(
                        TAG,
                        "Evicted LRU '$lruKey' (used ${entry.hitCount} times, " +
                                "${entry.sizeBytes / 1024 / 1024}MB)"
                )
            } catch (e: Exception) {
                Log.w(TAG, "Error closing evicted entry: ${e.message}")
            }
        }
    }

    /** Clear all cached models */
    @Synchronized
    fun clear() {
        Log.i(TAG, "Clearing cache (${cache.size} entries)")
        cache.values.forEach { entry ->
            try {
                entry.model.close()
            } catch (e: Exception) {
                Log.w(TAG, "Error closing cache entry: ${e.message}")
            }
        }
        cache.clear()
        totalCachedBytes = 0L
        hits = 0
        misses = 0
        evictions = 0
    }

    /** Remove specific entry from cache */
    @Synchronized
    fun remove(key: String): Boolean {
        val entry = cache.remove(key)
        if (entry != null) {
            totalCachedBytes -= entry.sizeBytes
            try {
                entry.model.close()
                Log.i(TAG, "Removed '$key' from cache")
                return true
            } catch (e: Exception) {
                Log.w(TAG, "Error closing removed entry: ${e.message}")
            }
        }
        return false
    }

    /** Get cache statistics */
    @Synchronized
    fun getStats(): CacheStats {
        val totalSize = totalCachedBytes / 1024 / 1024
        return CacheStats(
                entries = cache.size,
                totalSizeMB = totalSize,
                hits = hits,
                misses = misses,
                evictions = evictions
        )
    }

    /** Log current cache statistics */
    private fun logStats() {
        val stats = getStats()
        Log.d(TAG, stats.toString())
    }

    /** Get current cache size in MB */
    @Synchronized
    fun getCurrentSizeMB(): Long {
        return totalCachedBytes / 1024 / 1024
    }

    /** Check if key exists in cache */
    @Synchronized
    fun contains(key: String): Boolean {
        return cache.containsKey(key)
    }
}
