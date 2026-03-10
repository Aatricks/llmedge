package io.aatricks.llmedge.core

import android.content.Context
import io.aatricks.llmedge.runtime.ModelCache
import io.aatricks.llmedge.util.MemoryMetrics

internal object ModelCacheFactory {
    fun <T : AutoCloseable> create(
        context: Context,
        scope: LLMEdgeScope,
        maxCacheSize: Int,
        maxMemoryMB: Long,
    ): ModelCache<T> =
        ModelCache<T>(
            maxCacheSize = maxCacheSize,
            maxMemoryMB = maxMemoryMB,
            closeScope = scope.coroutineScope,
        ).apply {
            systemMemoryProvider = {
                MemoryMetrics.snapshot(context).availSystemMemBytes / (1024L * 1024L)
            }
        }
}