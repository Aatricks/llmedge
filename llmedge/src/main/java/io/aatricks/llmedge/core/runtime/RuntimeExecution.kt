package io.aatricks.llmedge.core.runtime

internal data class RuntimeExecutionContext<TRuntime : ManagedRuntime>(
    val runtime: TRuntime,
    val acquire: RuntimeAcquireResult<TRuntime>,
)

internal suspend fun <TSpec, TOptions, TRuntime : ManagedRuntime, TResult> RuntimePool<TSpec, TOptions, TRuntime>.executeWithRuntimeRetry(
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
