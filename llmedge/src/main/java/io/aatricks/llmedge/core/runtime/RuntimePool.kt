package io.aatricks.llmedge.core.runtime

import android.content.Context
import io.aatricks.llmedge.RuntimeCacheConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ModelCacheFactory
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ModelCache
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.sync.Mutex

internal typealias RuntimeAcquireResult<TRuntime> = RuntimeCoordinator.AcquireResult<TRuntime>

/**
 * Holds a runtime for a long-lived session. While the lease is open the backing cache
 * entry is pinned (LRU eviction skips it); closing the lease releases the pin — or closes
 * the runtime outright when the lease had to fall back to an uncached load. Idempotent.
 */
internal class RuntimeLease<TRuntime : ManagedRuntime>(
    val runtime: TRuntime,
    private val release: () -> Unit,
) : AutoCloseable {
    private val released = AtomicBoolean(false)

    override fun close() {
        if (released.compareAndSet(false, true)) {
            release()
        }
    }
}

internal class RuntimePool<TSpec, TOptions, TRuntime : ManagedRuntime>(
    internal val cache: ModelCache<TRuntime>,
    private val cacheKeyPrefix: (TSpec, TOptions) -> String,
    private val loadRuntime: suspend (TSpec, TOptions, ComputeBackend) -> TRuntime,
    private val activeBackend: (TRuntime) -> ComputeBackend,
    private val candidateRequest: (TOptions) -> BackendCandidateResolver.Request,
    loadMutex: Mutex = Mutex(),
) : AutoCloseable {
    internal val coordinator =
        RuntimeCoordinator(
            cache = cache,
            cacheKeyPrefix = cacheKeyPrefix,
            loadRuntime = loadRuntime,
            activeBackend = activeBackend,
            candidateRequest = candidateRequest,
            loadMutex = loadMutex,
        )

    suspend fun prepare(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = coordinator.acquire(spec, options)

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

    /**
     * Acquire a runtime and pin its cache entry until the returned lease is closed, so
     * cache pressure from other loads cannot close it mid-session. If the entry is evicted
     * between acquire and pin (or the runtime was never cached), falls back to a detached
     * load that the lease itself owns and closes.
     */
    suspend fun acquireLeased(
        spec: TSpec,
        options: TOptions,
    ): RuntimeLease<TRuntime> {
        repeat(2) {
            val result = coordinator.acquireDetailed(spec, options)
            val key = RuntimeCacheKeyBuilder.withBackend(result.keyPrefix, result.backend)
            if (cache.pin(key, result.runtime)) {
                return RuntimeLease(result.runtime) { cache.unpin(key) }
            }
        }
        val detached = coordinator.loadDetached(spec, options)
        return RuntimeLease(detached) { detached.close() }
    }

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

    suspend fun <T> executeWithRetry(
        spec: TSpec,
        options: TOptions,
        onRetry: ((RuntimeAcquireResult<TRuntime>, Throwable) -> Unit)? = null,
        execute: suspend (RuntimeExecutionContext<TRuntime>) -> T,
    ): T =
        coordinator.executeWithRuntimeRetry(
            spec = spec,
            options = options,
            onRetry = onRetry,
            execute = execute,
        )

    suspend fun <T> withExclusiveRuntime(
        runtime: TRuntime,
        dispatcher: CoroutineDispatcher? = null,
        execute: suspend (TRuntime) -> T,
    ): T =
        when (dispatcher) {
            null -> runtime.runExclusive(execute)
            else -> runtime.runExclusive(dispatcher, execute)
        }

    suspend fun <T> withExclusiveRuntimeRetry(
        spec: TSpec,
        options: TOptions,
        dispatcher: CoroutineDispatcher? = null,
        onRetry: ((RuntimeAcquireResult<TRuntime>, Throwable) -> Unit)? = null,
        execute: suspend (runtime: TRuntime, acquire: RuntimeAcquireResult<TRuntime>) -> T,
    ): T =
        executeWithRetry(
            spec = spec,
            options = options,
            onRetry = onRetry,
        ) { execution ->
            withExclusiveRuntime(
                runtime = execution.runtime,
                dispatcher = dispatcher,
            ) { runtime ->
                execute(runtime, execution.acquire)
            }
        }

    override fun close() {
        cache.clear()
    }
}

internal data class RuntimePoolProfile<TSpec, TOptions, TRuntime : ManagedRuntime>(
    val cacheConfig: RuntimeCacheConfig,
    val cacheKeyPrefix: (TSpec, TOptions) -> String,
    val loadRuntime: suspend (TSpec, TOptions, ComputeBackend) -> TRuntime,
    val activeBackend: (TRuntime) -> ComputeBackend,
    val candidateRequest: (TOptions) -> BackendCandidateResolver.Request,
)

internal fun <TSpec, TOptions, TRuntime : ManagedRuntime> runtimePoolProfile(
    cacheConfig: RuntimeCacheConfig,
    cacheKeyPrefix: (TSpec, TOptions) -> String,
    loadRuntime: suspend (TSpec, TOptions, ComputeBackend) -> TRuntime,
    activeBackend: (TRuntime) -> ComputeBackend,
    candidateRequest: (TOptions) -> BackendCandidateResolver.Request,
): RuntimePoolProfile<TSpec, TOptions, TRuntime> =
    RuntimePoolProfile(
        cacheConfig = cacheConfig,
        cacheKeyPrefix = cacheKeyPrefix,
        loadRuntime = loadRuntime,
        activeBackend = activeBackend,
        candidateRequest = candidateRequest,
    )

internal fun <TSpec, TOptions, TRuntime : ManagedRuntime> createCachedRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    profile: RuntimePoolProfile<TSpec, TOptions, TRuntime>,
): RuntimePool<TSpec, TOptions, TRuntime> =
    RuntimePool(
        cache =
            ModelCacheFactory.create(
                context = context,
                scope = scope,
                maxCacheSize = profile.cacheConfig.maxEntries,
                maxMemoryMB = profile.cacheConfig.maxMemoryMb,
            ),
        cacheKeyPrefix = profile.cacheKeyPrefix,
        loadRuntime = profile.loadRuntime,
        activeBackend = profile.activeBackend,
        candidateRequest = profile.candidateRequest,
    )
