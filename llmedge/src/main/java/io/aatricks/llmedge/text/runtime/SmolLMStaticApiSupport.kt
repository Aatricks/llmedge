package io.aatricks.llmedge.text.runtime

internal object SmolLMStaticApiSupport {
    val defaultBlockingBatchSize: Int
        get() = SmolLMCompanionSupport.defaultBlockingBatchSize

    fun isOpenClAvailable(nativeCheck: () -> Boolean): Boolean =
        SmolLMCompanionSupport.isOpenClAvailable(nativeCheck)

    fun isVulkanBackendAvailable(nativeCheck: () -> Boolean): Boolean =
        SmolLMCompanionSupport.isVulkanBackendAvailable(nativeCheck)

    fun overrideNativeBridgeForTests(provider: (SmolLM) -> SmolLM.NativeBridge) {
        SmolLMCompanionSupport.overrideNativeBridgeForTests(provider)
    }

    fun resetNativeBridgeForTests() {
        SmolLMCompanionSupport.resetNativeBridgeForTests()
    }

    fun overrideNativeLibrarySupportForTests(support: SmolLMNativeLibrarySupport) {
        SmolLMCompanionSupport.overrideNativeLibrarySupportForTests(support)
    }

    fun resetNativeLibrarySupportForTests() {
        SmolLMCompanionSupport.resetNativeLibrarySupportForTests()
    }

    fun currentNativeLibrarySupport(): SmolLMNativeLibrarySupport =
        SmolLMCompanionSupport.currentNativeLibrarySupport()

    fun createLoadedForTests(
        nativePtr: Long,
        useVulkan: Boolean = false,
        loadedParams: SmolLM.InferenceParams = SmolLM.InferenceParams(),
    ): SmolLM = SmolLMCompanionSupport.createLoadedForTests(nativePtr, useVulkan, loadedParams)

    fun logDebug(message: String) = SmolLMCompanionSupport.logDebug(message)

    fun logWarning(message: String) = SmolLMCompanionSupport.logWarning(message)
}
