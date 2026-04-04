package io.aatricks.llmedge.core

import android.content.Context
import kotlinx.coroutines.CoroutineScope

internal inline fun <T> createOwnedClient(
    context: Context,
    scope: CoroutineScope,
    promptThreads: Int,
    build: (ClientBootstrapContext) -> T,
): T = ClientBootstrap.createOwned(context, scope, promptThreads, build)

abstract class OwnedClient internal constructor(
    private val ownedBootstrap: ClientBootstrapContext?,
) : AutoCloseable {
    protected fun closeOwned(closeManagedResources: () -> Unit) {
        ClientBootstrap.close(ownedBootstrap, closeManagedResources)
    }
}
