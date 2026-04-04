package io.aatricks.llmedge.text

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.OwnedClient
import io.aatricks.llmedge.core.createOwnedClient
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

internal fun TextModelOptions.toInferenceParams(config: LLMEdgeConfig): SmolLM.InferenceParams =
    SmolLM.InferenceParams(
        minP = minP ?: config.text.minP,
        temperature = temperature ?: config.text.temperature,
        storeChats = false,
        contextSize = contextSize ?: config.text.contextSize,
        numThreads = numThreads ?: config.text.promptThreads,
        generationThreads = generationThreads ?: numThreads ?: config.text.generationThreads,
        useMmap = useMmap ?: config.text.useMmap,
        useMlock = useMlock ?: config.text.useMlock,
        useFlashAttn = useFlashAttention ?: config.text.useFlashAttention,
        thinkingMode = thinkingMode,
        reasoningBudget = reasoningBudget,
    )

class TextClient internal constructor(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val modelResolver: ModelRepository,
    private val ownedBootstrap: ClientBootstrapContext? = null,
) : OwnedClient(ownedBootstrap) {
    companion object {
        private const val LOG_TAG = "TextClient"
        /** Cap for chat state snapshots — skip snapshotting if state exceeds 64 MB. */
        private const val MAX_CHAT_STATE_BYTES = 64L * 1024L * 1024L

        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): TextClient =
            createOwnedClient(context, scope, config) { bootstrap ->
                TextClient(
                    context = bootstrap.appContext,
                    scope = bootstrap.edgeScope,
                    config = config,
                    modelResolver = modelRepository,
                    ownedBootstrap = bootstrap,
                )
            }
    }

    @Volatile
    private var lastGenerationMetrics: SmolLM.GenerationMetrics? = null

    private val runtimePool = createTextRuntimePool(context, scope, config, modelResolver)
    private val runtimeSession = TextRuntimeSession(scope, config, ::updateGenerationMetrics)
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

    /**
     * Chat-oriented completion that captures the KV-cache state after generation
     * so callers can restore it on the next turn, avoiding full re-tokenization.
     *
     * @param restoreState If non-null, restores this state before generation instead
     *                     of re-preparing the model from scratch.
     * @return Pair of (response text, state snapshot). The snapshot may be null if
     *         the native runtime does not support state capture or the state exceeds
     *         [maxStateBytes].
     */
    internal suspend fun chatTurn(
        runtime: ManagedTextModel,
        prompt: String,
        systemPrompt: String?,
        options: TextModelOptions,
        maxTokens: Int,
        batchSize: Int,
        restoreState: ByteArray? = null,
        maxStateBytes: Long = MAX_CHAT_STATE_BYTES,
    ): Pair<String, ByteArray?> =
        runtimeSession.chatTurn(
            runtime = runtime,
            prompt = prompt,
            systemPrompt = systemPrompt,
            options = options,
            maxTokens = maxTokens,
            batchSize = batchSize,
            restoreState = restoreState,
            maxStateBytes = maxStateBytes,
        )

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
