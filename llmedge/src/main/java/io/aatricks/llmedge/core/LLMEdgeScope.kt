package io.aatricks.llmedge.core

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel

class LLMEdgeScope(parentScope: CoroutineScope, inferenceThreads: Int) : AutoCloseable {
    private val supervisorJob = SupervisorJob(parentScope.coroutineContext[Job])
    private val inferenceExecutor: ExecutorService =
        Executors.newFixedThreadPool(inferenceThreads.coerceAtLeast(1)) { runnable ->
            Thread(runnable, "llmedge-inference").apply { isDaemon = true }
        }
    private val inferenceDispatcherInternal: ExecutorCoroutineDispatcher =
        inferenceExecutor.asCoroutineDispatcher()

    val coroutineScope: CoroutineScope =
        CoroutineScope(parentScope.coroutineContext + supervisorJob)
    val inferenceDispatcher: CoroutineDispatcher
        get() = inferenceDispatcherInternal
    val resources: ResourceScope = ResourceScope()

    override fun close() {
        var failure: Throwable? = null
        try {
            resources.close()
        } catch (t: Throwable) {
            failure = t
        }
        supervisorJob.cancel()
        inferenceDispatcherInternal.close()
        failure?.let { throw IllegalStateException("Failed to close LLMEdgeScope", it) }
    }
}
