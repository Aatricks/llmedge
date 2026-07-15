package io.aatricks.llmedge.image.ipc

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.image.DiffusionPhaseListener
import io.aatricks.llmedge.image.DiffusionRequestExecutor
import io.aatricks.llmedge.image.DefaultImageExecutionPlanSelector
import io.aatricks.llmedge.image.DefaultVideoExecutionPlanSelector
import io.aatricks.llmedge.image.GenerationStreamEvent
import io.aatricks.llmedge.image.ImageClientState
import io.aatricks.llmedge.image.ImageGenerationExecutor
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.image.VideoGenerationExecutor
import io.aatricks.llmedge.image.VideoGenerationRequest
import io.aatricks.llmedge.image.createDiffusionRuntimePool
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationTraceEvent
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.core.ProgressEvent
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex

/** The historical in-process diffusion stack, extracted verbatim from ImageClient. */
internal class InProcessDiffusionEngine(
    appContext: Context,
    private val edgeScope: LLMEdgeScope,
    config: LLMEdgeConfig,
    modelRepository: ModelRepository,
    logTag: String = "ImageClient",
    phaseListener: DiffusionPhaseListener? = null,
) : DiffusionEngine {
    private val runtimePool = createDiffusionRuntimePool(appContext, edgeScope, config, modelRepository, phaseListener)
    private val generationMutex = Mutex()
    private val imageRequestIds = AtomicLong(0L)
    private val state = ImageClientState()
    private val requestExecutor = DiffusionRequestExecutor(runtimePool, state, logTag)
    private val imageGenerationExecutor =
        ImageGenerationExecutor(
            config = config,
            generationMutex = generationMutex,
            imageRequestIds = imageRequestIds,
            state = state,
            requestExecutor = requestExecutor,
            executionPlanSelector = DefaultImageExecutionPlanSelector(appContext, modelRepository, phaseListener = phaseListener),
            logTag = logTag,
            phaseListener = phaseListener,
        )
    private val videoGenerationExecutor =
        VideoGenerationExecutor(
            scope = edgeScope,
            config = config,
            generationMutex = generationMutex,
            state = state,
            requestExecutor = requestExecutor,
            executionPlanSelector = DefaultVideoExecutionPlanSelector(appContext, modelRepository, phaseListener = phaseListener),
            phaseListener = phaseListener,
        )

    override suspend fun generate(params: ImageGenerationRequest): Bitmap = imageGenerationExecutor.generate(params)

    override fun generateStream(params: ImageGenerationRequest): Flow<GenerationStreamEvent> =
        callbackFlow {
            val job =
                edgeScope.coroutineScope.launch {
                    try {
                        val bitmap = imageGenerationExecutor.generate(params) { step, totalSteps ->
                            trySend(GenerationStreamEvent.Progress(ProgressEvent.Step("Sampling", step, totalSteps)))
                        }
                        trySend(GenerationStreamEvent.Completed(listOf(bitmap)))
                        close()
                    } catch (t: Throwable) {
                        close(t)
                    }
                }
            awaitClose {
                job.cancel()
                cancelGeneration()
            }
        }

    override fun generateVideo(params: VideoGenerationRequest): Flow<GenerationStreamEvent> =
        videoGenerationExecutor.generate(params)

    override fun cancelGeneration() {
        state.activeModel?.cancelGeneration()
    }

    override fun lastGenerationMetrics(): GenerationMetrics? = state.lastGenerationMetrics

    override fun lastImageRequestTraceForTests(): List<ImageGenerationTraceEvent> = state.lastImageRequestTrace

    override fun close() {
        cancelGeneration()
        state.activeModel = null
        runtimePool.close()
    }
}
