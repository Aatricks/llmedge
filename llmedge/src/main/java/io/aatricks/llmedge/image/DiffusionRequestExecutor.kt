package io.aatricks.llmedge.image

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.core.runtime.executeWithRuntimeRetry
import io.aatricks.llmedge.core.runtime.runExclusive
import io.aatricks.llmedge.image.diffusion.StableDiffusion

internal class DiffusionRequestExecutor(
    private val runtimePool: RuntimePool<DiffusionRuntimeSpec, DiffusionLoadOptions, ManagedDiffusionModel>,
    private val state: ImageClientState,
    private val logTag: String,
) {
    suspend fun <T> executeWithRuntimeRetry(
        spec: DiffusionRuntimeSpec,
        options: DiffusionLoadOptions,
        retryMessage: String,
        execute: suspend (AcquiredDiffusionRuntime) -> T,
    ): T =
        runtimePool.executeWithRuntimeRetry(
            spec = spec,
            options = options,
            onRetry = { _, _ -> AndroidLogAdapter.w(logTag, "$retryMessage '${spec.model.cacheKey}'") },
        ) { execution ->
            AndroidLogAdapter.i(
                logTag,
                "Runtime acquire completed: role=${spec.role} backend=${execution.acquire.backend} cacheHit=${execution.acquire.cacheHit} loadMs=${execution.acquire.modelLoadTimeMs}",
            )
            execute(AcquiredDiffusionRuntime(execution.runtime, execution.acquire))
        }

    suspend fun <T> withRuntimeModel(
        spec: DiffusionRuntimeSpec,
        options: DiffusionLoadOptions,
        retryMessage: String,
        execute: suspend (
            model: StableDiffusion,
            runtime: ManagedDiffusionModel,
            acquire: io.aatricks.llmedge.core.runtime.RuntimeAcquireResult<ManagedDiffusionModel>,
        ) -> T,
    ): T =
        executeWithRuntimeRetry(
            spec = spec,
            options = options,
            retryMessage = retryMessage,
        ) { acquired ->
            acquired.runtime.runExclusive { runtime ->
                withActiveModel(runtime.model) { model ->
                    execute(model, runtime, acquired.acquire)
                }
            }
        }

    suspend fun <T> withActiveModel(
        model: StableDiffusion,
        execute: suspend (StableDiffusion) -> T,
    ): T {
        state.activeModel = model
        AndroidLogAdapter.i(logTag, "withActiveModel enter: handle=${System.identityHashCode(model)}")
        return try {
            execute(model)
        } finally {
            model.clearImageRequestTrace()
            if (state.activeModel === model) {
                state.activeModel = null
            }
            AndroidLogAdapter.i(logTag, "withActiveModel exit: handle=${System.identityHashCode(model)}")
        }
    }
}

internal data class AcquiredDiffusionRuntime(
    val runtime: ManagedDiffusionModel,
    val acquire: io.aatricks.llmedge.core.runtime.RuntimeAcquireResult<ManagedDiffusionModel>,
)
