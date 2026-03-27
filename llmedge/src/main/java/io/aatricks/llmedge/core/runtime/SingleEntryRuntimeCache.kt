package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.core.AndroidLogAdapter

internal class SingleEntryRuntimeCache<K, T : AutoCloseable>(
    private val label: String,
) {
    private val entries = LinkedHashMap<K, T>(2, 0.75f, true)

    @Synchronized
    fun get(key: K): T? = entries[key]

    @Synchronized
    fun put(
        key: K,
        value: T,
    ) {
        if (entries.containsKey(key)) {
            entries.remove(key)?.closeQuietly()
        }
        while (entries.isNotEmpty()) {
            val lruKey = entries.keys.first()
            AndroidLogAdapter.i(label, "Evicting cached runtime for $lruKey")
            entries.remove(lruKey)?.closeQuietly()
        }
        entries[key] = value
    }

    @Synchronized
    fun clear() {
        entries.values.forEach { it.closeQuietly() }
        entries.clear()
    }

    private fun AutoCloseable.closeQuietly() {
        try {
            close()
        } catch (_: Exception) {
        }
    }
}
