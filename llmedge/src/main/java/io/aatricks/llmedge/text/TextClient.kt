package io.aatricks.llmedge.text

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.core.runtime.BackendFailureClassifier
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.tools.Tool
import io.aatricks.llmedge.tools.ToolAgent
import io.aatricks.llmedge.tools.ToolPolicies
import io.aatricks.llmedge.tools.ToolPolicy
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

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
    private val ownedScope: LLMEdgeScope? = null,
) : AutoCloseable {
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
        ): TextClient {
            val appContext = context.applicationContext
            val edgeScope = LLMEdgeScope(scope, config.text.promptThreads)
            return TextClient(appContext, edgeScope, config, modelRepository, ownedScope = edgeScope)
        }
    }

    @Volatile
    private var lastGenerationMetrics: SmolLM.GenerationMetrics? = null

    private val runtimePool = createTextRuntimePool(context, scope, config, modelResolver)

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
        runtimePool.acquire(model, options)
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
        lastGenerationMetrics = null
        return try {
            generateOnce(request)
        } catch (error: InferenceFailedException) {
            retryGenerateIfNeeded(request, error)
        }
    }

    private suspend fun generateOnce(request: TextGenerationRequest): String {
        val runtime = acquire(request.model, request.options)
        return try {
            complete(
                runtime = runtime,
                prompt = request.prompt,
                systemPrompt = request.systemPrompt,
                options = request.options,
                maxTokens = request.maxTokens,
                batchSize = request.batchSize,
            )
        } catch (error: InferenceFailedException) {
            recordBackendFailureIfNeeded(request.model, request.options, runtime, error)
            throw error
        }
    }

    private suspend fun retryGenerateIfNeeded(
        request: TextGenerationRequest,
        error: InferenceFailedException,
    ): String {
        if (isBackendFailure(error)) {
            AndroidLogAdapter.w(
                LOG_TAG,
                "Retrying text generation on the next backend after a backend-specific failure for '${request.model.cacheKey}'",
            )
            return try {
                generateOnce(request)
            } catch (retryError: InferenceFailedException) {
                retryError.addSuppressed(error)
                throw retryError
            }
        }

        val fallbackRequest = buildSafeRetryRequest(request) ?: throw error
        if (!isDecodeFailure(error)) {
            throw error
        }

        AndroidLogAdapter.w(
            LOG_TAG,
            "Retrying text generation with CPU-safe settings after decode failure for '${request.model.cacheKey}'",
        )
        invalidateRuntime(request.model, request.options)

        return try {
            generateOnce(fallbackRequest)
        } catch (retryError: InferenceFailedException) {
            retryError.addSuppressed(error)
            throw retryError
        }
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

    fun stream(request: TextGenerationRequest): Flow<TextStreamEvent> = flow {
        emit(TextStreamEvent.Started(request.prompt))
        val runtime = acquire(request.model, request.options)
        lastGenerationMetrics = null
        val response = StringBuilder()
        try {
            streamCompletion(runtime, request.prompt, request.systemPrompt, request.options, request.batchSize).collect { chunk ->
                response.append(chunk)
                emit(TextStreamEvent.Chunk(chunk))
            }
        } catch (error: InferenceFailedException) {
            recordBackendFailureIfNeeded(request.model, request.options, runtime, error)
            throw error
        }
        emit(TextStreamEvent.Completed(response.toString()))
    }

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
        return runtimePool.acquire(model, options)
    }

    internal suspend fun complete(
        runtime: ManagedTextModel,
        prompt: String,
        systemPrompt: String?,
        options: TextModelOptions,
        maxTokens: Int,
        batchSize: Int,
    ): String =
        runtime.mutex.withLock {
            withContext(scope.inferenceDispatcher) {
                runtime.ensureOpen()
                prepareModel(runtime.model, systemPrompt, options)
                try {
                    val effectiveBatchSize = resolveBatchSize(batchSize, maxTokens)
                    runtime.model.getResponse(prompt, maxTokens, effectiveBatchSize).also {
                        lastGenerationMetrics = runtime.model.getLastGenerationMetrics()
                    }
                } finally {
                    runtime.model.clearKvCache()
                }
            }
        }

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
        runtime.mutex.withLock {
            withContext(scope.inferenceDispatcher) {
                runtime.ensureOpen()
                if (restoreState != null) {
                    runtime.model.setStateBytes(restoreState)
                    runtime.model.setThinkingMode(options.thinkingMode)
                    options.reasoningBudget?.let(runtime.model::setReasoningBudget)
                } else {
                    prepareModel(runtime.model, systemPrompt, options)
                }
                val effectiveBatchSize = resolveBatchSize(batchSize, maxTokens)
                val response = runtime.model.getResponse(prompt, maxTokens, effectiveBatchSize)
                lastGenerationMetrics = runtime.model.getLastGenerationMetrics()
                val stateBytes = runtime.model.getStateBytes()?.takeIf { it.size <= maxStateBytes }
                runtime.model.clearKvCache()
                response to stateBytes
            }
        }

    internal fun streamCompletion(
        runtime: ManagedTextModel,
        prompt: String,
        systemPrompt: String?,
        options: TextModelOptions,
        batchSize: Int,
    ): Flow<String> =
        flow {
            runtime.mutex.withLock {
                withContext(scope.inferenceDispatcher) {
                    runtime.ensureOpen()
                    prepareModel(runtime.model, systemPrompt, options)
                }
                try {
                    val effectiveBatchSize = resolveStreamBatchSize(batchSize)
                    runtime.model
                        .getResponseAsFlow(prompt, scope.inferenceDispatcher, effectiveBatchSize)
                        .buffer(64)
                        .collect { chunk ->
                            if (chunk != "[EOG]") {
                                emit(chunk)
                            }
                        }
                    lastGenerationMetrics = runtime.model.getLastGenerationMetrics()
                } finally {
                    runtime.model.clearKvCache()
                }
            }
        }

    private fun prepareModel(
        model: SmolLM,
        systemPrompt: String?,
        options: TextModelOptions,
    ) {
        model.clearMessages()
        model.clearKvCache()
        systemPrompt?.takeUnless(String::isBlank)?.let(model::addSystemPrompt)
        model.setThinkingMode(options.thinkingMode)
        options.reasoningBudget?.let(model::setReasoningBudget)
    }

    private fun resolveStreamBatchSize(requestedBatchSize: Int): Int {
        val configuredBatchSize = config.text.streamBatchSize
        return when {
            requestedBatchSize == 0 -> configuredBatchSize
            requestedBatchSize > 0 -> requestedBatchSize
            else -> 1
        }
    }

    private fun resolveBatchSize(requestedBatchSize: Int, maxTokens: Int): Int {
        val configuredBatchSize = config.text.batchSize
        val preferredBatchSize =
            when {
                requestedBatchSize == 0 -> configuredBatchSize
                requestedBatchSize > 0 -> requestedBatchSize
                else -> 1
            }
        return if (maxTokens > 0) {
            minOf(preferredBatchSize, maxTokens.coerceAtLeast(1))
        } else {
            preferredBatchSize
        }
    }

    override fun close() {
        try {
            runtimePool.close()
        } finally {
            ownedScope?.close()
        }
    }

    private fun invalidateRuntime(model: ModelSpec, options: TextModelOptions) {
        runtimePool.invalidate(model, options)
    }

    private fun recordBackendFailureIfNeeded(
        model: ModelSpec,
        options: TextModelOptions,
        runtime: ManagedTextModel,
        error: InferenceFailedException,
    ) {
        val blacklisted = runtimePool.recordBackendFailureIfNeeded(model, options, runtime, error)
        if (!blacklisted) {
            return
        }
        val backend = runtime.model.getActiveBackend()
        AndroidLogAdapter.w(
            LOG_TAG,
            "Blacklisting $backend for text inference after a backend-specific failure on '${model.cacheKey}'",
        )
    }

    private fun buildSafeRetryRequest(request: TextGenerationRequest): TextGenerationRequest? {
        val effectiveUsesVulkan = request.options.useVulkan ?: config.text.useVulkan
        val effectiveUsesFlashAttention = request.options.useFlashAttention ?: config.text.useFlashAttention
        val effectiveBatchSize = resolveBatchSize(request.batchSize, request.maxTokens)

        if (!effectiveUsesVulkan && !effectiveUsesFlashAttention && effectiveBatchSize == 1) {
            return null
        }

        return request.copy(
            options = request.options.copy(useVulkan = false, useFlashAttention = false),
            batchSize = 1,
        )
    }

    private fun isDecodeFailure(error: InferenceFailedException): Boolean =
        error.message?.contains("llama_decode() failed") == true ||
            error.cause?.message?.contains("llama_decode() failed") == true

    private fun isBackendFailure(error: InferenceFailedException): Boolean =
        BackendFailureClassifier.isBackendFailure(error)

    internal suspend fun loadDetached(model: ModelSpec, options: TextModelOptions): ManagedTextModel =
        runtimePool.loadDetached(model, options)
}
