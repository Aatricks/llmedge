package io.aatricks.llmedge.text

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.ModelCache
import io.aatricks.llmedge.SmolLM
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.ModelCacheFactory
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Mutex
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
        minP = minP ?: config.defaultTextMinP,
        temperature = temperature ?: config.defaultTextTemperature,
        storeChats = false,
        contextSize = contextSize ?: config.defaultTextContextSize,
        numThreads = numThreads ?: config.defaultTextThreads.coerceAtLeast(1),
        generationThreads = generationThreads ?: numThreads ?: config.defaultTextGenerationThreads.coerceAtLeast(1),
        useMmap = useMmap ?: config.defaultUseMmap,
        useMlock = useMlock ?: config.defaultUseMlock,
        useFlashAttn = useFlashAttention ?: config.defaultUseFlashAttention,
        thinkingMode = thinkingMode,
        reasoningBudget = reasoningBudget,
    )

internal class ManagedTextModel(
    val fileSizeBytes: Long,
    val model: SmolLM,
) : AutoCloseable {
    val mutex: Mutex = Mutex()

    fun estimatedNativeMemoryBytes(): Long =
        maxOf(
            model.getEstimatedNativeMemoryBytes().takeIf { it > 0L } ?: 0L,
            fileSizeBytes + model.getEstimatedStateMemoryBytes().coerceAtLeast(0L),
            fileSizeBytes,
        )

    override fun close() {
        model.close()
    }
}

class TextClient internal constructor(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val modelResolver: ModelResolver,
) : AutoCloseable {
    private companion object {
        private const val LOG_TAG = "TextClient"
    }

    @Volatile
    private var lastGenerationMetrics: SmolLM.GenerationMetrics? = null

    private val cache =
        ModelCacheFactory.create<ManagedTextModel>(
            context = context,
            scope = scope,
            maxCacheSize = config.textCacheSize,
            maxMemoryMB = config.textCacheMemoryMb,
        )
    private val loadMutex = Mutex()

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
        acquire(model, options)
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
        return complete(
            runtime = runtime,
            prompt = request.prompt,
            systemPrompt = request.systemPrompt,
            options = request.options,
            maxTokens = request.maxTokens,
            batchSize = request.batchSize,
        )
    }

    private suspend fun retryGenerateIfNeeded(
        request: TextGenerationRequest,
        error: InferenceFailedException,
    ): String {
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
        streamCompletion(runtime, request.prompt, request.systemPrompt, request.options, request.batchSize).collect { chunk ->
            response.append(chunk)
            emit(TextStreamEvent.Chunk(chunk))
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

    internal suspend fun acquire(model: ModelSpec, options: TextModelOptions): ManagedTextModel {
        val key = buildCacheKey(model, options)
        cache.get(key)?.let { return it }

        return loadMutex.withLock {
            cache.get(key)?.let { return@withLock it }
            val modelFile = modelResolver.resolve(context, model)
            val smol = SmolLM(useVulkan = options.useVulkan ?: config.textUseVulkan)
            smol.load(modelFile.absolutePath, options.toInferenceParams(config))
            val runtime = ManagedTextModel(fileSizeBytes = modelFile.length(), model = smol)
            cache.put(
                key = key,
                model = runtime,
                sizeBytes = runtime.estimatedNativeMemoryBytes(),
                sizeProvider = runtime::estimatedNativeMemoryBytes,
            )
            runtime
        }
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
        val configuredBatchSize = config.defaultTextStreamBatchSize.coerceAtLeast(1)
        return when {
            requestedBatchSize == 0 -> configuredBatchSize
            requestedBatchSize > 0 -> requestedBatchSize
            else -> 1
        }
    }

    private fun resolveBatchSize(requestedBatchSize: Int, maxTokens: Int): Int {
        val configuredBatchSize = config.defaultTextBatchSize.coerceAtLeast(1)
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

    private fun buildCacheKey(model: ModelSpec, options: TextModelOptions): String =
        listOf(
            model.cacheKey,
            "ctx=${options.contextSize ?: config.defaultTextContextSize ?: 0L}",
            "threads=${options.numThreads ?: config.defaultTextThreads}",
            "genThreads=${options.generationThreads ?: options.numThreads ?: config.defaultTextGenerationThreads}",
            "mmap=${options.useMmap ?: config.defaultUseMmap}",
            "mlock=${options.useMlock ?: config.defaultUseMlock}",
            "flash=${options.useFlashAttention ?: config.defaultUseFlashAttention}",
            "vulkan=${options.useVulkan ?: config.textUseVulkan}",
        ).joinToString("|")

    override fun close() {
        cache.clear()
    }

    private fun invalidateRuntime(model: ModelSpec, options: TextModelOptions) {
        cache.remove(buildCacheKey(model, options))
    }

    private fun buildSafeRetryRequest(request: TextGenerationRequest): TextGenerationRequest? {
        val effectiveUsesVulkan = request.options.useVulkan ?: config.textUseVulkan
        val effectiveUsesFlashAttention = request.options.useFlashAttention ?: config.defaultUseFlashAttention
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
}
