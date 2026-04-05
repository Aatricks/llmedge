package io.aatricks.llmedge.text.runtime

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.NativeBridgeProvider
import io.aatricks.llmedge.core.NativeLibraryLoader
import io.aatricks.llmedge.core.NativeProbeSupport
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.CpuTopology

internal object SmolLMCompanionSupport {
    private const val LOG_TAG = "SmolLM"

    val defaultBlockingBatchSize: Int =
        CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.TOKEN_GENERATION)
            .coerceIn(4, 16)

    private val defaultNativeLibrarySupport =
        SmolLMNativeLibrarySupport {
            NativeLibraryLoader.ensureSmolLMLoaded(
                required = true,
                onDebug = { message -> AndroidLogAdapter.d(LOG_TAG, message) },
                onError = { message, throwable -> AndroidLogAdapter.e(LOG_TAG, message, throwable) },
            )
        }

    private val noOpNativeLibrarySupport = SmolLMNativeLibrarySupport { }

    @Volatile
    private var nativeLibrarySupportOverride: SmolLMNativeLibrarySupport? = null

    private val nativeBridgeProvider = NativeBridgeProvider(SmolLMNativeBridgeSupport.defaultProvider())

    fun createNativeBridge(instance: SmolLM): SmolLM.NativeBridge = nativeBridgeProvider.create(instance)

    fun currentNativeLibrarySupport(): SmolLMNativeLibrarySupport =
        nativeLibrarySupportOverride ?: defaultNativeLibrarySupport

    fun isOpenClAvailable(nativeCheck: () -> Boolean): Boolean =
        NativeProbeSupport.withLoadedOrDefault(
            defaultValue = false,
            ensureLoaded = currentNativeLibrarySupport()::ensureLoaded,
            probe = nativeCheck,
        )

    fun isVulkanBackendAvailable(nativeCheck: () -> Boolean): Boolean =
        NativeProbeSupport.withLoadedOrDefault(
            defaultValue = true,
            ensureLoaded = currentNativeLibrarySupport()::ensureLoaded,
            probe = nativeCheck,
        )

    fun overrideNativeBridgeForTests(provider: (SmolLM) -> SmolLM.NativeBridge) {
        nativeBridgeProvider.override(provider)
        nativeLibrarySupportOverride = noOpNativeLibrarySupport
    }

    fun resetNativeBridgeForTests() {
        nativeBridgeProvider.reset()
        nativeLibrarySupportOverride = null
    }

    fun overrideNativeLibrarySupportForTests(support: SmolLMNativeLibrarySupport) {
        nativeLibrarySupportOverride = support
    }

    fun resetNativeLibrarySupportForTests() {
        nativeLibrarySupportOverride = null
    }

    fun createLoadedForTests(
        nativePtr: Long,
        useVulkan: Boolean,
        loadedParams: SmolLM.InferenceParams,
    ): SmolLM {
        val model = SmolLM(useVulkan, nativeLibrarySupport = noOpNativeLibrarySupport)
        model.state.nativePtr = nativePtr
        model.state.loadedInferenceParams = loadedParams
        model.state.selectedBackend = if (useVulkan) ComputeBackend.VULKAN else ComputeBackend.CPU
        return model
    }

    fun logDebug(message: String) {
        AndroidLogAdapter.d(LOG_TAG, message)
    }

    fun logWarning(message: String) {
        AndroidLogAdapter.w(LOG_TAG, message)
    }
}
