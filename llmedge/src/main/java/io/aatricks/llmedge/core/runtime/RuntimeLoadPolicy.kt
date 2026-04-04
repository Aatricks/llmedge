package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.ComputeBackend

internal object RuntimeLoadPolicy {
    fun candidates(
        preferredBackend: ComputeBackend,
        includeCpuFallback: Boolean,
    ): List<ComputeBackend> {
        if (!includeCpuFallback || preferredBackend == ComputeBackend.CPU) {
            return listOf(preferredBackend)
        }

        return listOf(preferredBackend, ComputeBackend.CPU)
    }

    fun candidates(
        request: BackendCandidateResolver.Request,
        preferredBackend: ComputeBackend? = null,
        includeCpuFallback: Boolean = preferredBackend == null,
    ): List<ComputeBackend> {
        if (preferredBackend == null) {
            return BackendCandidateResolver.candidates(request)
        }

        return candidates(preferredBackend, includeCpuFallback)
    }

    fun recordBackendFailureIfNeeded(
        request: BackendCandidateResolver.Request,
        backend: ComputeBackend,
        preferredBackend: ComputeBackend? = null,
        blacklistGpuFailures: Boolean = preferredBackend == null,
    ): Boolean {
        if (!blacklistGpuFailures || preferredBackend != null || backend == ComputeBackend.CPU) {
            return false
        }

        BackendCandidateResolver.blacklist(request.subsystem, backend)
        return request.subsystem != null
    }
}
