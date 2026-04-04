package io.aatricks.llmedge.core.runtime

import kotlinx.coroutines.CoroutineDispatcher

internal class ManagedRuntimeExecutor<TSpec, TOptions, TRuntime : ManagedRuntime>(
    private val runtimePool: RuntimePool<TSpec, TOptions, TRuntime>,
    private val dispatcher: CoroutineDispatcher? = null,
) {
    suspend fun prepare(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = runtimePool.coordinator.acquire(spec, options)

    suspend fun acquire(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = runtimePool.coordinator.acquire(spec, options)

    suspend fun acquireDetailed(
        spec: TSpec,
        options: TOptions,
    ): RuntimeAcquireResult<TRuntime> = runtimePool.coordinator.acquireDetailed(spec, options)

    suspend fun loadDetached(
        spec: TSpec,
        options: TOptions,
    ): TRuntime = runtimePool.coordinator.loadDetached(spec, options)

    fun invalidate(
        spec: TSpec,
        options: TOptions,
    ) {
        runtimePool.coordinator.invalidate(spec, options)
    }

    fun recordBackendFailureIfNeeded(
        spec: TSpec,
        options: TOptions,
        runtime: TRuntime,
        error: Throwable,
    ): Boolean = runtimePool.coordinator.recordBackendFailureIfNeeded(spec, options, runtime, error)

    suspend fun <T> executeWithRetry(
        spec: TSpec,
        options: TOptions,
        onRetry: ((RuntimeAcquireResult<TRuntime>, Throwable) -> Unit)? = null,
        execute: suspend (RuntimeExecutionContext<TRuntime>) -> T,
    ): T =
        runtimePool.coordinator.executeWithRuntimeRetry(
            spec = spec,
            options = options,
            onRetry = onRetry,
            execute = execute,
        )

    suspend fun <T> withExclusiveRuntime(
        runtime: TRuntime,
        execute: suspend (TRuntime) -> T,
    ): T =
        when (val currentDispatcher = dispatcher) {
            null -> runtime.runExclusive(execute)
            else -> runtime.runExclusive(currentDispatcher, execute)
        }

    suspend fun <T> withExclusiveRuntimeRetry(
        spec: TSpec,
        options: TOptions,
        onRetry: ((RuntimeAcquireResult<TRuntime>, Throwable) -> Unit)? = null,
        execute: suspend (runtime: TRuntime, acquire: RuntimeAcquireResult<TRuntime>) -> T,
    ): T =
        executeWithRetry(
            spec = spec,
            options = options,
            onRetry = onRetry,
        ) { execution ->
            withExclusiveRuntime(execution.runtime) { runtime ->
                execute(runtime, execution.acquire)
            }
        }

    fun close() {
        runtimePool.close()
    }
}
