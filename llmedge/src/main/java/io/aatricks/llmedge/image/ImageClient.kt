package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.OwnedClient
import io.aatricks.llmedge.core.createOwnedClient
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationTraceEvent
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.sync.Mutex

data class ImageGenerationRequest(
    val prompt: String,
    val negative: String = "",
    val width: Int = 512,
    val height: Int = 512,
    val steps: Int = 20,
    val cfgScale: Float = 7.0f,
    val seed: Long = -1L,
    val flashAttention: Boolean = true,
    val forceSequentialLoad: Boolean = false,
    val easyCache: EasyCacheParams = EasyCacheParams(),
    val loraModelDir: String? = null,
    val loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
    val model: ModelSpec? = null,
)

data class VideoGenerationRequest(
    val prompt: String,
    val negative: String = "",
    val width: Int = 512,
    val height: Int = 512,
    val videoFrames: Int = 16,
    val steps: Int = 20,
    val cfgScale: Float = 7.0f,
    val seed: Long = -1L,
    val flowShift: Float = Float.POSITIVE_INFINITY,
    val flashAttention: Boolean = true,
    val forceSequentialLoad: Boolean = false,
    val initImage: Bitmap? = null,
    val strength: Float = 1.0f,
    val sampleMethod: SampleMethod = SampleMethod.DEFAULT,
    val scheduler: Scheduler = Scheduler.DEFAULT,
    val easyCache: EasyCacheParams = EasyCacheParams(),
    val loraModelDir: String? = null,
    val loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
    val taehv: ModelSpec? = null,
    val model: ModelSpec? = null,
    val vae: ModelSpec? = null,
    val textEncoder: ModelSpec? = null,
) {
    fun actualFrameCount(): Int = (videoFrames - 1) / 4 * 4 + 1
}

class ImageClient internal constructor(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val resolver: ModelRepository,
    private val ownedBootstrap: ClientBootstrapContext? = null,
) : OwnedClient(ownedBootstrap) {
    companion object {
        private const val LOG_TAG = "ImageClient"

        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): ImageClient =
            createOwnedClient(context, scope, config) { bootstrap ->
                ImageClient(
                    context = bootstrap.appContext,
                    scope = bootstrap.edgeScope,
                    config = config,
                    resolver = modelRepository,
                    ownedBootstrap = bootstrap,
                )
            }

        internal fun resetVideoVulkanBlacklistForTests() {
            BackendRuntimePolicy.resetForTests()
        }
    }

    private val runtimePool = createDiffusionRuntimePool(context, scope, config, resolver)
    private val generationMutex = Mutex()
    private val imageRequestIds = AtomicLong(0L)
    private val state = ImageClientState()
    private val requestExecutor = DiffusionRequestExecutor(runtimePool, state, LOG_TAG)
    private val imageGenerationExecutor =
        ImageGenerationExecutor(
            config = config,
            generationMutex = generationMutex,
            imageRequestIds = imageRequestIds,
            state = state,
            requestExecutor = requestExecutor,
            logTag = LOG_TAG,
        )
    private val videoGenerationExecutor =
        VideoGenerationExecutor(
            scope = scope,
            config = config,
            generationMutex = generationMutex,
            state = state,
            requestExecutor = requestExecutor,
        )

    /**
     * Generate a single bitmap from text.
     *
     * This client resolves and loads the requested model for the operation, while advanced callers
     * can use [StableDiffusion] directly when they need to hold a warmed runtime across requests.
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution, loading, or native
     * generation fails.
     */
    suspend fun generate(
        params: ImageGenerationRequest,
    ): Bitmap = imageGenerationExecutor.generate(params)

    /**
     * Stream progress and final frames for text-to-video generation.
     *
     * Cancel the returned flow collection to stop the active generation.
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution, loading, or native
     * generation fails.
     */
    fun generateVideo(
        params: VideoGenerationRequest,
    ): Flow<GenerationStreamEvent> = videoGenerationExecutor.generate(params)

    /** Request cancellation for the active generation, if any. */
    fun cancelGeneration() {
        state.activeModel?.cancelGeneration()
    }

    fun getLastGenerationMetrics(): GenerationMetrics? = state.lastGenerationMetrics

    internal fun getLastImageRequestTraceForTests(): List<ImageGenerationTraceEvent> = state.lastImageRequestTrace

    override fun close() {
        closeOwned {
            cancelGeneration()
            state.activeModel = null
            runtimePool.close()
        }
    }
}
