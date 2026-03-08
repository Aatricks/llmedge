package io.aatricks.llmedge.core

internal class NativeBridgeProvider<TInstance, TBridge>(
    private val defaultProvider: (TInstance) -> TBridge,
) {
    @Volatile
    private var provider: (TInstance) -> TBridge = defaultProvider

    fun create(instance: TInstance): TBridge = provider(instance)

    fun override(provider: (TInstance) -> TBridge) {
        this.provider = provider
    }

    fun reset() {
        provider = defaultProvider
    }
}