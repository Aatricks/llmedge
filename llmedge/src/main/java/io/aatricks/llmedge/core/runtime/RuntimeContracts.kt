package io.aatricks.llmedge.core.runtime

internal fun interface RuntimeKeyStrategy<TSpec, TOptions> {
    fun prefix(spec: TSpec, options: TOptions): String
}

internal fun interface RuntimeLoader<TSpec, TOptions, TRuntime : ManagedRuntime> {
    suspend fun load(spec: TSpec, options: TOptions): TRuntime
}

internal fun interface BackendPolicy<TOptions> {
    fun request(options: TOptions): BackendCandidateResolver.Request
}
