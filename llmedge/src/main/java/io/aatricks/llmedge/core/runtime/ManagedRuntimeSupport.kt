package io.aatricks.llmedge.core.runtime

import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

internal abstract class ManagedRuntimeBase : ManagedRuntime {
    final override val mutex: Mutex = Mutex()
    private val closed = AtomicBoolean(false)

    protected fun ensureOpen(name: String) {
        check(!closed.get()) { "$name has been closed" }
    }

    protected fun closeOnce(closeAction: () -> Unit) {
        if (!closed.compareAndSet(false, true)) {
            return
        }
        runBlocking {
            mutex.withLock {
                closeAction()
            }
        }
    }
}
