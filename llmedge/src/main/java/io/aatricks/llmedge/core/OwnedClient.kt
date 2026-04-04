package io.aatricks.llmedge.core

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import kotlinx.coroutines.CoroutineScope

internal inline fun <T> createOwnedClient(
    context: Context,
    scope: CoroutineScope,
    config: LLMEdgeConfig,
    build: (ClientBootstrapContext) -> T,
): T = ClientBootstrap.createOwned(context, scope, config.execution.inferenceThreads, build)

abstract class OwnedClient internal constructor(
    private val ownedBootstrap: ClientBootstrapContext?,
) : AutoCloseable {
    protected fun closeOwned(closeManagedResources: () -> Unit) {
        ClientBootstrap.close(ownedBootstrap, closeManagedResources)
    }
}
