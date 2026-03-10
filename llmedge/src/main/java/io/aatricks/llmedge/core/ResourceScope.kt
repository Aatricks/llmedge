package io.aatricks.llmedge.core

class ResourceScope : AutoCloseable {
    private val resources = LinkedHashSet<AutoCloseable>()
    private val lock = Any()

    fun <T : AutoCloseable> register(resource: T): T {
        synchronized(lock) {
            resources.add(resource)
        }
        return resource
    }

    fun unregister(resource: AutoCloseable) {
        synchronized(lock) {
            resources.remove(resource)
        }
    }

    override fun close() {
        val snapshot =
            synchronized(lock) {
                val reversed = resources.toList().asReversed()
                resources.clear()
                reversed
            }

        var failure: Throwable? = null
        snapshot.forEach { resource ->
            try {
                resource.close()
            } catch (t: Throwable) {
                if (failure == null) {
                    failure = t
                } else {
                    failure?.addSuppressed(t)
                }
            }
        }

        failure?.let { throw IllegalStateException("Failed to close one or more resources", it) }
    }
}
