package io.aatricks.llmedge.core.runtime

import android.content.Context
import io.aatricks.llmedge.RuntimeCacheConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ModelCacheFactory
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ModelCache
import kotlinx.coroutines.sync.Mutex

internal typealias RuntimeAcquireResult<TRuntime> = RuntimeCoordinator.AcquireResult<TRuntime>

internal class RuntimePool<TSpec, TOptions, TRuntime : ManagedRuntime>(
    private val cache: ModelCache<TRuntime>,
    private val cacheKeyPrefix: (TSpec, TOptions) -> String,
    private val loadRuntime: suspend (TSpec, TOptions, ComputeBackend) -> TRuntime,
    private val activeBackend: (TRuntime) -> ComputeBackend,
    private val candidateRequest: (TOptions) -> BackendCandidateResolver.Request,
    loadMutex: Mutex = Mutex(),
) : AutoCloseable {
    private val coordinator =
        RuntimeCoordinator(
            cache = cache,
            cacheKeyPrefix = cacheKeyPrefix,
            loadRuntime = loadRuntime,
            activeBackend = activeBackend,
            candidateRequest = candidateRequest,
            loadMutex = loadMutex,
        )

    suspend fun acquire(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = coordinator.acquire(spec, options)

    suspend fun acquireDetailed(
        spec: TSpec,
        options: TOptions,
    ): RuntimeAcquireResult<TRuntime> = coordinator.acquireDetailed(spec, options)

    suspend fun loadDetached(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = coordinator.loadDetached(spec, options)

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

internal fun <TSpec, TOptions, TRuntime : ManagedRuntime> createCachedRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    cacheConfig: RuntimeCacheConfig,
    cacheKeyPrefix: (TSpec, TOptions) -> String,
    loadRuntime: suspend (TSpec, TOptions, ComputeBackend) -> TRuntime,
    activeBackend: (TRuntime) -> ComputeBackend,
    candidateRequest: (TOptions) -> BackendCandidateResolver.Request,
): RuntimePool<TSpec, TOptions, TRuntime> =
    RuntimePool(
        cache =
            ModelCacheFactory.create(
                context = context,
                scope = scope,
                maxCacheSize = cacheConfig.maxEntries,
                maxMemoryMB = cacheConfig.maxMemoryMb,
            ),
        cacheKeyPrefix = cacheKeyPrefix,
        loadRuntime = loadRuntime,
        activeBackend = activeBackend,
        candidateRequest = candidateRequest,
    )
