package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.DiffusionWorkerMode
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.core.FeatureContext
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.OwnedFeatureClient
import io.aatricks.llmedge.core.featureClientFactory
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationTraceEvent
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.ipc.DiffusionEngine
import io.aatricks.llmedge.image.ipc.InProcessDiffusionEngine
import io.aatricks.llmedge.image.ipc.IsolatedDiffusionEngine
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow

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
    // Split-model image generation (FLUX.2 Klein): [model] is the diffusion transformer,
    // [vae] the autoencoder, [textEncoder] the Qwen3 LLM encoder. When [splitDiffusionModel] is
    // true the runtime routes [model] -> diffusion_model_path and [textEncoder] -> llm_path.
    val vae: ModelSpec? = null,
    val textEncoder: ModelSpec? = null,
    val t5xxl: ModelSpec? = null,
    val clipL: ModelSpec? = null,
    val clipG: ModelSpec? = null,
    val clipVision: ModelSpec? = null,
    val llmVision: ModelSpec? = null,
    val controlNet: ModelSpec? = null,
    val photoMaker: ModelSpec? = null,
    val embeddingsConnectors: ModelSpec? = null,
    // Standalone DiT checkpoints (for example MiniT2I) are routed through
    // diffusion_model_path while [textEncoder] remains in the T5 slot.
    val diffusionModelOnly: Boolean = false,
    val splitDiffusionModel: Boolean = false,
    // Null chooses direct or staged execution from the resolved component sizes and device headroom.
    // True forces staged execution; false forces direct execution and disables automatic retry.
    val sequential: Boolean? = null,
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
    /** True forces staged loading; false chooses direct or staged loading from resolved sizes and current memory headroom. */
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
    val highNoiseDiffusionModel: ModelSpec? = null,
) {
    fun actualFrameCount(): Int = (videoFrames - 1) / 4 * 4 + 1
}

class ImageClient internal constructor(
    featureContext: FeatureContext,
    private val ownedBootstrap: ClientBootstrapContext? = null,
) : OwnedFeatureClient(featureContext, ownedBootstrap) {
    companion object {
        private const val LOG_TAG = "ImageClient"
        private val FACTORY = featureClientFactory(::ImageClient)

        @Deprecated(
            message = "Prefer LLMEdge.create(...).image in new app code. This factory remains available for advanced construction and tests.",
        )
        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): ImageClient = FACTORY.create(context, scope, config, modelRepository, Unit)

        @JvmSynthetic
        internal fun forTesting(
            context: Context,
            scope: LLMEdgeScope,
            config: LLMEdgeConfig,
            resolver: ModelRepository,
            ownedBootstrap: ClientBootstrapContext? = null,
        ): ImageClient =
            FACTORY.forTesting(context, scope, config, resolver, Unit, ownedBootstrap)

        internal fun resetVideoVulkanBlacklistForTests() {
            BackendRuntimePolicy.resetForTests()
        }

        /**
         * Clears persisted backend hang verdicts (recorded by the isolated-worker watchdog) and
         * the in-memory blacklist, re-enabling GPU backends on the next load. Verdicts also clear
         * automatically when the OS fingerprint changes (system/driver update).
         */
        @JvmStatic
        fun resetBackendVerdicts(context: Context) {
            io.aatricks.llmedge.image.ipc.BackendVerdictStore(context).reset()
            BackendRuntimePolicy.resetForTests()
        }
    }

    init {
        // Seed persisted hang verdicts into the in-memory blacklist for BOTH modes, so an
        // in-process client on a device with a recorded GPU hang also avoids the broken backend.
        if (config.image.persistBackendVerdicts) {
            BackendRuntimePolicy.seed(io.aatricks.llmedge.image.ipc.BackendVerdictStore(appContext).load())
        }
    }

    private val engine: DiffusionEngine =
        when (config.image.workerMode) {
            DiffusionWorkerMode.IN_PROCESS ->
                InProcessDiffusionEngine(appContext, edgeScope, config, modelRepository, LOG_TAG)
            DiffusionWorkerMode.ISOLATED_PROCESS ->
                IsolatedDiffusionEngine(appContext, edgeScope, config)
        }

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
    ): Bitmap = engine.generate(params)

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
    ): Flow<GenerationStreamEvent> = engine.generateVideo(params)

    /** Request cancellation for the active generation, if any. */
    fun cancelGeneration() {
        engine.cancelGeneration()
    }

    fun getLastGenerationMetrics(): GenerationMetrics? = engine.lastGenerationMetrics()

    internal fun getLastImageRequestTraceForTests(): List<ImageGenerationTraceEvent> = engine.lastImageRequestTraceForTests()

    override fun close() {
        closeOwned {
            engine.close()
        }
    }
}
