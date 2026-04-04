package io.aatricks.llmedge.runtime

import java.util.concurrent.ConcurrentHashMap

internal class BackendBlacklistRegistry {
    private val blacklistedBackends = ConcurrentHashMap<Pair<ComputeSubsystem, ComputeBackend>, Boolean>()

    fun reset() {
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

internal class NativeLibraryRegistry {
    private val loadedLibraries = mutableSetOf<String>()

    @Synchronized
    fun loadOnce(
        name: String,
        load: (String) -> Unit,
    ) {
        if (loadedLibraries.contains(name)) {
            return
        }
        try {
            load(name)
            loadedLibraries += name
        } catch (error: Throwable) {
            loadedLibraries -= name
            throw error
        }
    }
}

internal data class RuntimeEnvironment(
    val backendBlacklistRegistry: BackendBlacklistRegistry = BackendBlacklistRegistry(),
    val nativeLibraryRegistry: NativeLibraryRegistry = NativeLibraryRegistry(),
)

internal object RuntimeEnvironmentHolder {
    @Volatile
    private var environment: RuntimeEnvironment = RuntimeEnvironment()

    fun current(): RuntimeEnvironment = environment

    fun resetForTests() {
        environment = RuntimeEnvironment()
    }
}
