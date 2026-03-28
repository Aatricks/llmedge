package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ModelCache
import kotlinx.coroutines.sync.Mutex

internal class RuntimePool<TSpec, TOptions, TRuntime : ManagedRuntime>(
    private val cache: ModelCache<TRuntime>,
    private val keyStrategy: RuntimeKeyStrategy<TSpec, TOptions>,
    private val runtimeLoader: RuntimeLoader<TSpec, TOptions, TRuntime>,
    private val activeBackend: (TRuntime) -> ComputeBackend,
    private val backendPolicy: BackendPolicy<TOptions>,
    loadMutex: Mutex = Mutex(),
) : AutoCloseable {
    private val coordinator =
        RuntimeCoordinator(
            cache = cache,
            cacheKeyPrefix = keyStrategy::prefix,
            loadRuntime = runtimeLoader::load,
            activeBackend = activeBackend,
            candidateRequest = backendPolicy::request,
            loadMutex = loadMutex,
        )

    suspend fun acquire(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = coordinator.acquire(spec, options)

    suspend fun loadDetached(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = runtimeLoader.load(spec, options)

    fun invalidate(
        spec: TSpec,
        options: TOptions,
    ) {
        coordinator.invalidate(spec, options)
    }

    fun recordBackendFailureIfNeeded(
        spec: TSpec,
        options: TOptions,
        runtime: TRuntime,
        error: Throwable,
    ): Boolean = coordinator.recordBackendFailureIfNeeded(spec, options, runtime, error)

    override fun close() {
        cache.clear()
    }
}
