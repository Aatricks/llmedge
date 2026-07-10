package io.aatricks.llmedge.core.runtime

import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ModelCache
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

internal class RuntimeCoordinator<TSpec, TOptions, TRuntime : ManagedRuntime>(
    private val cache: ModelCache<TRuntime>,
    private val cacheKeyPrefix: (TSpec, TOptions) -> String,
    private val loadRuntime: suspend (TSpec, TOptions, ComputeBackend) -> TRuntime,
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
        val request = candidateRequest(options)
        findCachedRuntime(prefix, request)?.let { runtime ->
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
            findCachedRuntime(prefix, request)?.let { runtime ->
                return@withLock AcquireResult(
                    runtime = runtime,
                    cacheHit = true,
                    keyPrefix = prefix,
                    backend = activeBackend(runtime),
                    acquireTimeMs = elapsedMillis(startNanos),
                    modelLoadTimeMs = 0L,
                )
            }
            loadAcrossCandidates(
                spec = spec,
                options = options,
                prefix = prefix,
                request = request,
                startNanos = startNanos,
                cacheLoadedRuntime = true,
            )
        }
    }

    suspend fun loadDetached(
        spec: TSpec,
        options: TOptions,
    ): TRuntime =
        loadAcrossCandidates(
            spec = spec,
            options = options,
            prefix = cacheKeyPrefix(spec, options),
            request = candidateRequest(options),
            startNanos = System.nanoTime(),
            cacheLoadedRuntime = false,
        ).runtime

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
        RuntimeLoadPolicy.recordBackendFailureIfNeeded(request, backend)
        cache.remove(RuntimeCacheKeyBuilder.withBackend(cacheKeyPrefix(spec, options), backend))
        return true
    }

    private fun findCachedRuntime(
        prefix: String,
        request: BackendCandidateResolver.Request,
    ): TRuntime? {
        for (backend in RuntimeLoadPolicy.candidates(request)) {
            cache.get(RuntimeCacheKeyBuilder.withBackend(prefix, backend))?.let { return it }
        }
        return null
    }

    private suspend fun loadAcrossCandidates(
        spec: TSpec,
        options: TOptions,
        prefix: String,
        request: BackendCandidateResolver.Request,
        startNanos: Long,
        cacheLoadedRuntime: Boolean,
    ): AcquireResult<TRuntime> {
        var lastError: Throwable? = null
        for (backendCandidate in RuntimeLoadPolicy.candidates(request)) {
            val loadStartNanos = System.nanoTime()
            try {
                val runtime = loadRuntime(spec, options, backendCandidate)
                val backend = activeBackend(runtime)
                if (cacheLoadedRuntime) {
                    cache.put(
                        key = RuntimeCacheKeyBuilder.withBackend(prefix, backend),
                        model = runtime,
                        sizeBytes = runtime.estimatedSizeBytes(),
                        sizeProvider = runtime::estimatedSizeBytes,
                    )
                }
                return AcquireResult(
                    runtime = runtime,
                    cacheHit = false,
                    keyPrefix = prefix,
                    backend = backend,
                    acquireTimeMs = elapsedMillis(startNanos),
                    modelLoadTimeMs = elapsedMillis(loadStartNanos),
                )
            } catch (error: Throwable) {
                if (error is kotlinx.coroutines.CancellationException) {
                    throw error
                }
                lastError = error
                if (backendCandidate == ComputeBackend.CPU) {
                    throw error
                }
                if (BackendFailureClassifier.isBackendFailure(error)) {
                    RuntimeLoadPolicy.recordBackendFailureIfNeeded(request, backendCandidate)
                }
            }
        }
        throw lastError ?: IllegalStateException("Runtime load failed without a reported cause")
    }

    private fun elapsedMillis(startNanos: Long): Long =
        ((System.nanoTime() - startNanos) / 1_000_000L).coerceAtLeast(0L)
}
