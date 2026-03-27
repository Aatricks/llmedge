package io.aatricks.llmedge.runtime

import java.util.concurrent.ConcurrentHashMap

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
    WHISPER,
}

internal object BackendRuntimePolicy {
    private val blacklistedBackends = ConcurrentHashMap<Pair<ComputeSubsystem, ComputeBackend>, Boolean>()

    fun resetForTests() {
        blacklistedBackends.clear()
    }

    fun blacklist(
        subsystem: ComputeSubsystem,
        backend: ComputeBackend,
    ) {
        if (backend == ComputeBackend.CPU) {
            return
        }
        blacklistedBackends[subsystem to backend] = true
    }

    fun isBlacklisted(
        subsystem: ComputeSubsystem,
        backend: ComputeBackend,
    ): Boolean = backend != ComputeBackend.CPU && blacklistedBackends[subsystem to backend] == true

    fun candidates(
        subsystem: ComputeSubsystem,
        allowGpu: Boolean,
        openClAvailable: Boolean,
        vulkanAvailable: Boolean,
    ): List<ComputeBackend> {
        if (!allowGpu) {
            return listOf(ComputeBackend.CPU)
        }

        val result = ArrayList<ComputeBackend>(3)
        if (openClAvailable && !isBlacklisted(subsystem, ComputeBackend.OPENCL)) {
            result += ComputeBackend.OPENCL
        }
        if (vulkanAvailable && !isBlacklisted(subsystem, ComputeBackend.VULKAN)) {
            result += ComputeBackend.VULKAN
        }
        result += ComputeBackend.CPU
        return result
    }
}
