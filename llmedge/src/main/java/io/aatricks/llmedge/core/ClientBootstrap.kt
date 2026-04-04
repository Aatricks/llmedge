package io.aatricks.llmedge.core

import android.content.Context
import kotlinx.coroutines.CoroutineScope

internal class ClientBootstrapContext private constructor(
    val appContext: Context,
    val edgeScope: LLMEdgeScope,
) : AutoCloseable {
    companion object {
        fun create(
            context: Context,
            scope: CoroutineScope,
            promptThreads: Int,
        ): ClientBootstrapContext =
            ClientBootstrapContext(
                appContext = context.applicationContext,
                edgeScope = LLMEdgeScope(scope, promptThreads),
            )
    }

    override fun close() {
        edgeScope.close()
    }
}

internal object ClientBootstrap {
    fun create(
        context: Context,
        scope: CoroutineScope,
        promptThreads: Int,
    ): ClientBootstrapContext = ClientBootstrapContext.create(context, scope, promptThreads)

    inline fun <T> createOwned(
        context: Context,
        scope: CoroutineScope,
        promptThreads: Int,
        build: (ClientBootstrapContext) -> T,
    ): T {
        val bootstrap = create(context, scope, promptThreads)
        return build(bootstrap)
    }

    fun close(owner: ClientBootstrapContext?) {
        owner?.close()
    }

    inline fun close(
        owner: ClientBootstrapContext?,
        closeManagedResources: () -> Unit,
    ) {
        try {
            closeManagedResources()
        } finally {
            close(owner)
        }
    }
}
