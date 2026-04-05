package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.runtime.ComputeBackend

internal inline fun <T> runBackendAttempts(
    candidates: Iterable<ComputeBackend>,
    onFailure: (ComputeBackend, Throwable?) -> Unit = { _, _ -> },
    noinline exhaustedError: ((Throwable?) -> Throwable)? = null,
    attempt: (ComputeBackend) -> T?,
): T? {
    var lastError: Throwable? = null
    for (backend in candidates) {
        try {
            attempt(backend)?.let { return it }
            onFailure(backend, null)
        } catch (bindingError: NativeBindingException) {
            throw bindingError
        } catch (error: Throwable) {
            lastError = error
            onFailure(backend, error)
        }
    }
    exhaustedError?.let { throw it(lastError) }
    return null
}
