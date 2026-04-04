package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

internal class VisionPipeline(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val resolver: ModelRepository,
    private val config: LLMEdgeConfig,
    private val smolLmFactory: (Boolean) -> SmolLM = { useVulkan -> SmolLM(useVulkan = useVulkan) },
    private val projectorFactory: () -> Projector = { Projector() },
) : AutoCloseable {
    private companion object {
        private const val TAG = "VisionPipeline"
        private const val JPEG_QUALITY = 90
    }

    private val runtimePool =
        createVisionRuntimePool(
            context = context,
            scope = scope,
            resolver = resolver,
            config = config,
            smolLmFactory = smolLmFactory,
            projectorFactory = projectorFactory,
        )
    private val inputPreparer = VisionInputPreparer(context, JPEG_QUALITY)
    private val runtimeExecutor = VisionRuntimeExecutor(context)

    suspend fun prepare(
        model: ModelSpec,
        projector: ModelSpec,
        numThreads: Int,
        generationThreads: Int,
        onStatus: ((String) -> Unit)? = null,
    ) {
        withContext(Dispatchers.IO) {
            acquireRuntime(
                model = model,
                projector = projector,
                numThreads = numThreads,
                generationThreads = generationThreads,
            )
            AndroidLogAdapter.d(
                TAG,
                "prepare completed for ${model.cacheKey}",
            )
        }
    }

    suspend fun analyze(
        request: VisionRequest,
        onStatus: ((String) -> Unit)? = null,
    ): VisionPipelineResult =
        withContext(Dispatchers.IO) {
            val runtime =
                acquireRuntime(
                    model = request.model,
                    projector = request.projector,
                    numThreads = request.numThreads,
                    generationThreads = request.generationThreads,
                )
            runtime.mutex.withLock {
                val preparedInput =
                    inputPreparer.prepare(
                        request = request,
                        runtime = runtime,
                        onStatus = onStatus,
                        logStage = ::logStage,
                    )
                try {
                    runtimeExecutor.execute(
                        request = request,
                        runtime = runtime,
                        preparedInput = preparedInput,
                        onStatus = onStatus,
                        logStage = ::logStage,
                    )
                } finally {
                    preparedInput.close()
                }
            }
        }

    private suspend fun acquireRuntime(
        model: ModelSpec,
        projector: ModelSpec,
        numThreads: Int?,
        generationThreads: Int?,
    ): ManagedVisionRuntime {
        val loadStartedNs = System.nanoTime()
        val runtime =
            runtimePool.acquire(
                VisionRuntimeSpec(model = model, projector = projector),
                VisionLoadOptions(
                    numThreads = (numThreads ?: config.text.promptThreads).coerceAtLeast(1),
                    generationThreads = (generationThreads ?: numThreads ?: config.text.generationThreads).coerceAtLeast(1),
                ),
            )
        logStage("runtime", "acquire", loadStartedNs)
        return runtime
    }

    private fun logStage(operation: String, stage: String, startedNs: Long) {
        val elapsedMs = (System.nanoTime() - startedNs) / 1_000_000
        AndroidLogAdapter.d(TAG, "$operation.$stage completed in ${elapsedMs}ms")
    }

    override fun close() {
        runtimePool.close()
    }
}
