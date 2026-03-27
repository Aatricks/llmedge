package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.ComputeBackend

internal object RuntimeCacheKeyBuilder {
    fun prefix(vararg parts: Any?): String = parts.joinToString("|")

    fun withBackend(prefix: String, backend: ComputeBackend): String =
        "$prefix|backend=${backend.name}"
}
