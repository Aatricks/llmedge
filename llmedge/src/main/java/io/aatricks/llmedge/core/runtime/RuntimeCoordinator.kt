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
    data class AcquireResult<TRuntime : ManagedRuntime>(
        val runtime: TRuntime,
        val cacheHit: Boolean,
        val keyPrefix: String,
        val backend: ComputeBackend,
        val acquireTimeMs: Long,
        val modelLoadTimeMs: Long,
    )

    suspend fun acquire(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = acquireDetailed(spec, options).runtime

    suspend fun acquireDetailed(
        spec: TSpec,
        options: TOptions,
    ): AcquireResult<TRuntime> {
        val prefix = cacheKeyPrefix(spec, options)
        val startNanos = System.nanoTime()
        findCachedRuntime(prefix, options)?.let { runtime ->
            return AcquireResult(
                runtime = runtime,
                cacheHit = true,
                keyPrefix = prefix,
                backend = activeBackend(runtime),
                acquireTimeMs = elapsedMillis(startNanos),
                modelLoadTimeMs = 0L,
            )
        }

        return loadMutex.withLock {
            findCachedRuntime(prefix, options)?.let { runtime ->
                return@withLock AcquireResult(
                    runtime = runtime,
                    cacheHit = true,
                    keyPrefix = prefix,
                    backend = activeBackend(runtime),
                    acquireTimeMs = elapsedMillis(startNanos),
                    modelLoadTimeMs = 0L,
                )
            }
            val loadStartNanos = System.nanoTime()
            val runtime = loadRuntime(spec, options)
            val backend = activeBackend(runtime)
            cache.put(
                key = RuntimeCacheKeyBuilder.withBackend(prefix, backend),
                model = runtime,
                sizeBytes = runtime.estimatedSizeBytes(),
                sizeProvider = runtime::estimatedSizeBytes,
            )
            AcquireResult(
                runtime = runtime,
                cacheHit = false,
                keyPrefix = prefix,
                backend = backend,
                acquireTimeMs = elapsedMillis(startNanos),
                modelLoadTimeMs = elapsedMillis(loadStartNanos),
            )
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

    private fun elapsedMillis(startNanos: Long): Long =
        ((System.nanoTime() - startNanos) / 1_000_000L).coerceAtLeast(0L)
}
