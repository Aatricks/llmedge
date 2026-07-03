package io.aatricks.llmedge.text

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.core.FeatureContext
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.OwnedFeatureClient
import io.aatricks.llmedge.core.featureClientFactory
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.tools.Tool
import io.aatricks.llmedge.tools.ToolAgent
import io.aatricks.llmedge.tools.ToolPolicies
import io.aatricks.llmedge.tools.ToolPolicy
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow

data class TextModelOptions(
    val contextSize: Long? = null,
    val chatTemplate: String? = null,
    val numThreads: Int? = null,
    val generationThreads: Int? = null,
    val minP: Float? = null,
    val temperature: Float? = null,
    val useMmap: Boolean? = null,
    val useMlock: Boolean? = null,
    val useFlashAttention: Boolean? = null,
    val thinkingMode: SmolLM.ThinkingMode = SmolLM.ThinkingMode.DEFAULT,
    val reasoningBudget: Int? = null,
    val useVulkan: Boolean? = null,
)

data class TextGenerationRequest(
    val prompt: String,
    val model: ModelSpec,
    val systemPrompt: String? = null,
    val options: TextModelOptions = TextModelOptions(),
    val maxTokens: Int = -1,
    val batchSize: Int = 0,
)

internal fun TextModelOptions.toInferenceParams(
    config: LLMEdgeConfig,
    fallbackChatTemplate: String? = null,
): SmolLM.InferenceParams =
    SmolLM.InferenceParams(
        minP = minP ?: config.text.minP,
        temperature = temperature ?: config.text.temperature,
        storeChats = false,
        contextSize = contextSize ?: config.text.contextSize,
        chatTemplate = chatTemplate ?: fallbackChatTemplate,
        numThreads = numThreads ?: config.text.promptThreads,
        generationThreads = generationThreads ?: numThreads ?: config.text.generationThreads,
        useMmap = useMmap ?: config.text.useMmap,
        useMlock = useMlock ?: config.text.useMlock,
        useFlashAttn = useFlashAttention ?: config.text.useFlashAttention,
        thinkingMode = thinkingMode,
        reasoningBudget = reasoningBudget,
    )

class TextClient internal constructor(
    featureContext: FeatureContext,
    private val ownedBootstrap: ClientBootstrapContext? = null,
) : OwnedFeatureClient(featureContext, ownedBootstrap) {
    companion object {
        private const val LOG_TAG = "TextClient"
        private val FACTORY = featureClientFactory(::TextClient)

        @Deprecated(
            message = "Prefer LLMEdge.create(...).text in new app code. This factory remains available for advanced construction and tests.",
        )
        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): TextClient = FACTORY.create(context, scope, config, modelRepository, Unit)

        @JvmSynthetic
        internal fun forTesting(
            context: Context,
            scope: LLMEdgeScope,
            config: LLMEdgeConfig,
            modelResolver: ModelRepository,
            ownedBootstrap: ClientBootstrapContext? = null,
        ): TextClient =
            FACTORY.forTesting(context, scope, config, modelResolver, Unit, ownedBootstrap)
    }

    @Volatile
    private var lastGenerationMetrics: SmolLM.GenerationMetrics? = null

    private val runtimePool = createTextRuntimePool(appContext, edgeScope, config, modelRepository)
    private val runtimeSession = TextRuntimeSession(edgeScope, config, ::updateGenerationMetrics)
    private val requestExecutor =
        TextRequestExecutor(
            runtimePool = runtimePool,
            runtimeSession = runtimeSession,
            config = config,
            logTag = LOG_TAG,
            resetMetrics = { updateGenerationMetrics(null) },
        )

    /**
     * Preload a text model into the cache so later generation calls avoid the initial model-load
     * cost on the calling path.
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution or loading fails.
     */
    suspend fun prepare(
        model: ModelSpec = config.models.text,
        options: TextModelOptions = TextModelOptions(),
    ) {
        requestExecutor.prepare(model, options)
    }

    /**
     * Generate a complete text response for [prompt].
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution, loading, or native
     * inference fails.
     */
    suspend fun generate(
        prompt: String,
        model: ModelSpec = config.models.text,
        systemPrompt: String? = null,
        options: TextModelOptions = TextModelOptions(),
        maxTokens: Int = -1,
        batchSize: Int = 0,
    ): String =
        generate(
            TextGenerationRequest(
                prompt = prompt,
                model = model,
                systemPrompt = systemPrompt,
                options = options,
                maxTokens = maxTokens,
                batchSize = batchSize,
            ),
        )

    suspend fun generate(request: TextGenerationRequest): String {
        return requestExecutor.generate(request)
    }

    /**
     * Stream a text response for [prompt].
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution, loading, or native
     * inference fails.
     */
    fun stream(
        prompt: String,
        model: ModelSpec = config.models.text,
        systemPrompt: String? = null,
        options: TextModelOptions = TextModelOptions(),
        batchSize: Int = 0,
    ): Flow<TextStreamEvent> =
        stream(
            TextGenerationRequest(
                prompt = prompt,
                model = model,
                systemPrompt = systemPrompt,
                options = options,
                batchSize = batchSize,
            ),
        )

    fun stream(request: TextGenerationRequest): Flow<TextStreamEvent> = requestExecutor.stream(request)

    fun getLastGenerationMetrics(): SmolLM.GenerationMetrics? = lastGenerationMetrics

    /**
     * Create a Kotlin-managed multi-turn chat session backed by this client.
     *
     * Use [ChatSession.prepare] if you want to preload the model before the first reply.
     */
    fun session(
        model: ModelSpec = config.models.text,
        memory: ConversationWindow = ConversationWindow(),
        systemPrompt: String? = null,
        options: TextModelOptions = TextModelOptions(),
    ): ChatSession = ChatSession(this, model, memory, systemPrompt, options)

    /**
     * Create a Kotlin-managed tool-calling agent backed by this client.
     *
     * The returned [ToolAgent] replays transcript state in Kotlin and requires explicit policy
     * approval before action tools execute.
     */
    fun toolAgent(
        tools: List<Tool>,
        model: ModelSpec = config.models.text,
        memory: ConversationWindow = ConversationWindow(),
        systemPrompt: String? = null,
        options: TextModelOptions = TextModelOptions(),
        policy: ToolPolicy = ToolPolicies.DENY_ACTIONS,
    ): ToolAgent = ToolAgent(this, tools, model, memory, systemPrompt, options, policy)

    internal suspend fun acquire(model: ModelSpec, options: TextModelOptions): ManagedTextModel {
        return requestExecutor.acquire(model, options)
    }

    internal suspend fun complete(
        runtime: ManagedTextModel,
        prompt: String,
        systemPrompt: String?,
        options: TextModelOptions,
        maxTokens: Int,
        batchSize: Int,
    ): String =
        runtimeSession.complete(runtime, prompt, systemPrompt, options, maxTokens, batchSize)

    internal fun streamCompletion(
        runtime: ManagedTextModel,
        prompt: String,
        systemPrompt: String?,
        options: TextModelOptions,
        batchSize: Int,
    ): Flow<String> =
        runtimeSession.streamCompletion(runtime, prompt, systemPrompt, options, batchSize)

    override fun close() {
        closeOwned {
            runtimePool.close()
        }
    }

    internal suspend fun loadDetached(model: ModelSpec, options: TextModelOptions): ManagedTextModel =
        requestExecutor.loadDetached(model, options)

    private fun updateGenerationMetrics(metrics: SmolLM.GenerationMetrics?) {
        lastGenerationMetrics = metrics
    }
}
