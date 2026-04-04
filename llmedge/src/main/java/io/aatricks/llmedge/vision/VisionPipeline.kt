package io.aatricks.llmedge.vision

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.FeatureContext
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal class VisionPipeline(
    private val featureContext: FeatureContext,
    private val smolLmFactory: (Boolean) -> SmolLM = { useVulkan -> SmolLM(useVulkan = useVulkan) },
    private val projectorFactory: () -> Projector = { Projector() },
) : AutoCloseable {
    internal constructor(
        context: android.content.Context,
        scope: io.aatricks.llmedge.core.LLMEdgeScope,
        resolver: io.aatricks.llmedge.model.ModelRepository,
        config: io.aatricks.llmedge.LLMEdgeConfig,
        smolLmFactory: (Boolean) -> SmolLM = { useVulkan -> SmolLM(useVulkan = useVulkan) },
        projectorFactory: () -> Projector = { Projector() },
    ) : this(
        featureContext =
            FeatureContext(
                appContext = context,
                edgeScope = scope,
                config = config,
                modelRepository = resolver,
            ),
        smolLmFactory = smolLmFactory,
        projectorFactory = projectorFactory,
    )

    private companion object {
        private const val TAG = "VisionPipeline"
        private const val JPEG_QUALITY = 90
    }

    private val runtimePool =
        createVisionRuntimePool(
            context = featureContext.appContext,
            scope = featureContext.edgeScope,
            resolver = featureContext.modelRepository,
            config = featureContext.config,
            smolLmFactory = smolLmFactory,
            projectorFactory = projectorFactory,
        )
    private val inputPreparer = VisionInputPreparer(featureContext.appContext, JPEG_QUALITY)
    private val pipelineExecutor = VisionRuntimeExecutor()

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
            runtimePool.withExclusiveRuntime(runtime) {
                val preparedInput =
                    inputPreparer.prepare(
                        request = request,
                        runtime = runtime,
                        onStatus = onStatus,
                        logStage = ::logStage,
                    )
                try {
                    pipelineExecutor.execute(
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
                    numThreads = (numThreads ?: featureContext.config.vision.promptThreads).coerceAtLeast(1),
                    generationThreads =
                        (generationThreads ?: numThreads ?: featureContext.config.vision.generationThreads).coerceAtLeast(1),
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
