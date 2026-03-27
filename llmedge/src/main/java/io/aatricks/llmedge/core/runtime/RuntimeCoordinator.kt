package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ModelCache
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

internal class RuntimeCoordinator<TSpec, TOptions, TRuntime : ManagedRuntime>(
    private val cache: ModelCache<TRuntime>,
    private val cacheKeyPrefix: (TSpec, TOptions) -> String,
    private val loadRuntime: suspend (TSpec, TOptions) -> TRuntime,
    private val activeBackend: (TRuntime) -> ComputeBackend,
    private val candidateRequest: (TOptions) -> BackendCandidateResolver.Request,
    private val loadMutex: Mutex = Mutex(),
) {
    suspend fun acquire(
        spec: TSpec,
        options: TOptions,
    ): TRuntime {
        val prefix = cacheKeyPrefix(spec, options)
        findCachedRuntime(prefix, options)?.let { return it }

        return loadMutex.withLock {
            findCachedRuntime(prefix, options)?.let { return@withLock it }
            val runtime = loadRuntime(spec, options)
            cache.put(
                key = RuntimeCacheKeyBuilder.withBackend(prefix, activeBackend(runtime)),
                model = runtime,
                sizeBytes = runtime.estimatedSizeBytes(),
                sizeProvider = runtime::estimatedSizeBytes,
            )
            runtime
        }
    }

    fun invalidate(
        spec: TSpec,
        options: TOptions,
    ) {
        val prefix = cacheKeyPrefix(spec, options)
        ComputeBackend.entries.forEach { backend ->
            cache.remove(RuntimeCacheKeyBuilder.withBackend(prefix, backend))
        }
    }

    fun recordBackendFailureIfNeeded(
        spec: TSpec,
        options: TOptions,
        runtime: TRuntime,
        error: Throwable,
    ): Boolean {
        if (!BackendFailureClassifier.isBackendFailure(error)) {
            return false
        }

        val backend = activeBackend(runtime)
        if (backend == ComputeBackend.CPU) {
            return false
        }

        val request = candidateRequest(options)
        BackendCandidateResolver.blacklist(request.subsystem, backend)
        cache.remove(RuntimeCacheKeyBuilder.withBackend(cacheKeyPrefix(spec, options), backend))
        return true
    }

    private fun findCachedRuntime(
        prefix: String,
        options: TOptions,
    ): TRuntime? {
        val request = candidateRequest(options)
        for (backend in BackendCandidateResolver.candidates(request)) {
            cache.get(RuntimeCacheKeyBuilder.withBackend(prefix, backend))?.let { return it }
        }
        return null
    }
}
