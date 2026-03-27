package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem

internal object BackendCandidateResolver {
    data class Request(
        val subsystem: ComputeSubsystem?,
        val allowGpu: Boolean,
        val openClAvailable: Boolean,
        val vulkanAvailable: Boolean,
    )

    fun candidates(request: Request): List<ComputeBackend> {
        val subsystem = request.subsystem ?: return listOf(ComputeBackend.CPU)
        return BackendRuntimePolicy.candidates(
            subsystem = subsystem,
            allowGpu = request.allowGpu,
            openClAvailable = request.openClAvailable,
            vulkanAvailable = request.vulkanAvailable,
        )
    }

    fun blacklist(
        subsystem: ComputeSubsystem?,
        backend: ComputeBackend,
    ) {
        if (subsystem == null) {
            return
        }
        BackendRuntimePolicy.blacklist(subsystem, backend)
    }
}
