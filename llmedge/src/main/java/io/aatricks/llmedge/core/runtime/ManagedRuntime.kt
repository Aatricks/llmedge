package io.aatricks.llmedge.core.runtime

import kotlinx.coroutines.sync.Mutex

internal interface ManagedRuntime : AutoCloseable {
    val mutex: Mutex

    fun estimatedSizeBytes(): Long
}
