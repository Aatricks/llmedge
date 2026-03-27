package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.ComputeBackend

internal object BackendExecutionRunner {
    suspend fun <T> run(
        candidates: List<ComputeBackend>,
        failureMessage: String,
        onBackendFailure: (ComputeBackend, Throwable) -> Unit,
        execute: suspend (ComputeBackend) -> T,
    ): T {
        var lastError: Throwable? = null
        for (backend in candidates) {
            try {
                return execute(backend)
            } catch (t: Throwable) {
                lastError = t
                if (backend != ComputeBackend.CPU) {
                    onBackendFailure(backend, t)
                    continue
                }
                throw t
            }
        }
        throw lastError ?: IllegalStateException(failureMessage)
    }
}
