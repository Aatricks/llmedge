package io.aatricks.llmedge.core.runtime

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

internal data class RuntimeExecutionContext<TRuntime : ManagedRuntime>(
    val runtime: TRuntime,
    val acquire: RuntimeAcquireResult<TRuntime>,
)

internal suspend inline fun <TRuntime : ManagedRuntime, TResult> TRuntime.runExclusive(
    dispatcher: CoroutineDispatcher,
    crossinline execute: suspend (TRuntime) -> TResult,
): TResult =
    mutex.withLock {
        withContext(dispatcher) {
            execute(this@runExclusive)
        }
    }

internal suspend inline fun <TRuntime : ManagedRuntime, TResult> TRuntime.runExclusive(
    crossinline execute: suspend (TRuntime) -> TResult,
): TResult =
    mutex.withLock {
        execute(this@runExclusive)
    }

internal suspend fun <TSpec, TOptions, TRuntime : ManagedRuntime, TResult> RuntimeCoordinator<TSpec, TOptions, TRuntime>.executeWithRuntimeRetry(
    spec: TSpec,
    options: TOptions,
    onRetry: ((RuntimeAcquireResult<TRuntime>, Throwable) -> Unit)? = null,
    execute: suspend (RuntimeExecutionContext<TRuntime>) -> TResult,
): TResult {
    while (true) {
        val acquire = acquireDetailed(spec, options)
        try {
            return execute(RuntimeExecutionContext(runtime = acquire.runtime, acquire = acquire))
        } catch (error: Throwable) {
            val blacklisted = recordBackendFailureIfNeeded(spec, options, acquire.runtime, error)
            if (!blacklisted) {
                throw error
            }
            onRetry?.invoke(acquire, error)
        }
    }
}
