package io.aatricks.llmedge.runtime

internal object ModelCacheBudgetPolicy {
    fun shouldEvict(
        entryCount: Int,
        maxCacheSize: Int,
        totalCachedBytes: Long,
        newSizeBytes: Long,
        maxMemoryMB: Long,
        systemMemoryProvider: (() -> Long)?,
    ): Boolean {
        if (entryCount >= maxCacheSize) {
            return true
        }

        val currentMemoryMB = totalCachedBytes / 1024 / 1024
        val newMemoryMB = currentMemoryMB + (newSizeBytes / 1024 / 1024)
        return newMemoryMB > effectiveMaxMemoryMb(maxMemoryMB, systemMemoryProvider)
    }

    fun shouldLogOversizedInsert(
        resolvedSizeBytes: Long,
        maxMemoryMB: Long,
    ): Boolean = resolvedSizeBytes > maxMemoryMB * 1024L * 1024L

    private fun effectiveMaxMemoryMb(
        maxMemoryMB: Long,
        systemMemoryProvider: (() -> Long)?,
    ): Long =
        systemMemoryProvider?.let { provider ->
            val avail = provider()
            val reserved = (avail * 0.1).toLong()
            val budget = (avail - reserved).coerceAtMost(maxMemoryMB)
            val minBudget = (maxMemoryMB / 4).coerceAtLeast(256L)
            budget.coerceAtLeast(minBudget)
        } ?: maxMemoryMB
}
