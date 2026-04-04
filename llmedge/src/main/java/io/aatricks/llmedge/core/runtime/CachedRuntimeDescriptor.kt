package io.aatricks.llmedge.core.runtime

import android.content.Context
import io.aatricks.llmedge.RuntimeCacheConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ModelCacheFactory
import io.aatricks.llmedge.runtime.ComputeBackend

internal data class CachedRuntimeDescriptor<TSpec, TOptions, TRuntime : ManagedRuntime>(
    val cache: RuntimeCacheConfig,
    val keyStrategy: RuntimeKeyStrategy<TSpec, TOptions>,
    val runtimeLoader: RuntimeLoader<TSpec, TOptions, TRuntime>,
    val activeBackend: (TRuntime) -> ComputeBackend,
    val backendPolicy: BackendPolicy<TOptions>,
)

internal fun <TSpec, TOptions, TRuntime : ManagedRuntime> createCachedRuntimePool(
    context: Context,
    scope: LLMEdgeScope,
    descriptor: CachedRuntimeDescriptor<TSpec, TOptions, TRuntime>,
): RuntimePool<TSpec, TOptions, TRuntime> =
    RuntimePool(
        cache =
            ModelCacheFactory.create(
                context = context,
                scope = scope,
                maxCacheSize = descriptor.cache.maxEntries,
                maxMemoryMB = descriptor.cache.maxMemoryMb,
            ),
        keyStrategy = descriptor.keyStrategy,
        runtimeLoader = descriptor.runtimeLoader,
        activeBackend = descriptor.activeBackend,
        backendPolicy = descriptor.backendPolicy,
    )
