package io.aatricks.llmedge.runtime

internal enum class ComputeBackend(val id: Int) {
    CPU(0),
    OPENCL(1),
    VULKAN(2),
    ;

    companion object {
        fun fromId(id: Int): ComputeBackend = entries.firstOrNull { it.id == id } ?: CPU
    }
}

internal enum class ComputeSubsystem {
    IMAGE,
    VIDEO,
    TEXT,
    VISION,
    WHISPER,
}

internal object BackendRuntimePolicy {
    private val registry: BackendBlacklistRegistry
        get() = RuntimeEnvironmentHolder.current().backendBlacklistRegistry

    fun resetForTests() {
        registry.reset()
    }

    fun blacklist(
        subsystem: ComputeSubsystem,
        backend: ComputeBackend,
    ) {
        registry.blacklist(subsystem, backend)
    }

    fun isBlacklisted(
        subsystem: ComputeSubsystem,
        backend: ComputeBackend,
    ): Boolean = registry.isBlacklisted(subsystem, backend)

    fun candidates(
        subsystem: ComputeSubsystem,
        allowGpu: Boolean,
        openClAvailable: Boolean,
        vulkanAvailable: Boolean,
    ): List<ComputeBackend> =
        registry.candidates(
            subsystem = subsystem,
            allowGpu = allowGpu,
            openClAvailable = openClAvailable,
            vulkanAvailable = vulkanAvailable,
        )
}
