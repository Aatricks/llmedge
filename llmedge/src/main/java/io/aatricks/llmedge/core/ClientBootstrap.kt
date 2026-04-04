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

    fun close(owner: ClientBootstrapContext?) {
        owner?.close()
    }
}
