package io.aatricks.llmedge.text

import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.runtime.BackendFailureClassifier
import io.aatricks.llmedge.core.runtime.RuntimePool
import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow

internal class TextRequestExecutor(
    private val runtimePool: RuntimePool<ModelSpec, TextModelOptions, ManagedTextModel>,
    private val runtimeSession: TextRuntimeSession,
    private val config: LLMEdgeConfig,
    private val logTag: String,
    private val resetMetrics: () -> Unit,
) {
    suspend fun prepare(
        model: ModelSpec,
        options: TextModelOptions,
    ) {
        runtimePool.prepare(model, options)
    }

    suspend fun generate(request: TextGenerationRequest): String {
        resetMetrics()
        return try {
            generateWithRuntimeRetry(request)
        } catch (error: InferenceFailedException) {
            retryGenerateIfNeeded(request, error)
        } catch (raced: IllegalStateException) {
            // A cached runtime can be evicted and closed between acquire() and the
            // mutex-guarded execution. Nothing ran yet in that case, so re-acquire once
            // (the pool will load a fresh runtime) instead of surfacing a raw internal error.
            if (!isClosedRuntimeError(raced)) throw raced
            AndroidLogAdapter.w(
                logTag,
                "Text runtime was evicted before use; re-acquiring once for '${request.model.cacheKey}'",
            )
            generateWithRuntimeRetry(request)
        }
    }

    fun stream(request: TextGenerationRequest): Flow<TextStreamEvent> =
            flow {
                emit(TextStreamEvent.Started(request.prompt))
            var runtime = runtimePool.acquire(request.model, request.options)
            resetMetrics()
            val response = StringBuilder()
            try {
                try {
                    streamInto(runtime, request, response)
                } catch (raced: IllegalStateException) {
                    // Eviction race: the runtime was closed between acquire() and use.
                    // Nothing has been emitted from the model yet, so retry once.
                    if (!isClosedRuntimeError(raced) || response.isNotEmpty()) throw raced
                    AndroidLogAdapter.w(
                        logTag,
                        "Text runtime was evicted before use; re-acquiring once for '${request.model.cacheKey}'",
                    )
                    runtime = runtimePool.acquire(request.model, request.options)
                    streamInto(runtime, request, response)
                }
            } catch (error: InferenceFailedException) {
                recordBackendFailureIfNeeded(request.model, request.options, runtime, error)
                throw error
            }
            emit(TextStreamEvent.Completed(response.toString()))
        }

    private suspend fun kotlinx.coroutines.flow.FlowCollector<TextStreamEvent>.streamInto(
        runtime: ManagedTextModel,
        request: TextGenerationRequest,
        response: StringBuilder,
    ) {
        runtimeSession
            .streamCompletion(
                runtime = runtime,
                prompt = request.prompt,
                systemPrompt = request.systemPrompt,
                options = request.options,
                batchSize = request.batchSize,
            ).collect { chunk ->
                response.append(chunk)
                emit(TextStreamEvent.Chunk(chunk))
            }
    }

    private fun isClosedRuntimeError(error: IllegalStateException): Boolean =
        error.message?.endsWith("has been closed") == true

    suspend fun acquire(
        model: ModelSpec,
        options: TextModelOptions,
    ): ManagedTextModel = runtimePool.acquire(model, options)

    fun invalidate(
        model: ModelSpec,
        options: TextModelOptions,
    ) {
        runtimePool.invalidate(model, options)
    }

    suspend fun loadDetached(
        model: ModelSpec,
        options: TextModelOptions,
    ): ManagedTextModel = runtimePool.loadDetached(model, options)

    private suspend fun generateWithRuntimeRetry(request: TextGenerationRequest): String =
        runtimePool.executeWithRetry(
            spec = request.model,
            options = request.options,
            onRetry = { _, _ ->
                AndroidLogAdapter.w(
                    logTag,
                    "Retrying text generation on the next backend after a backend-specific failure for '${request.model.cacheKey}'",
                )
            },
        ) { execution ->
            runtimeSession.complete(
                runtime = execution.runtime,
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
        if (!isDecodeFailure(error) || isBackendFailure(error)) {
            throw error
        }

        AndroidLogAdapter.w(
            logTag,
            "Retrying text generation with CPU-safe settings after decode failure for '${request.model.cacheKey}'",
        )
        invalidate(request.model, request.options)

        return try {
            generateWithRuntimeRetry(fallbackRequest)
        } catch (retryError: InferenceFailedException) {
            retryError.addSuppressed(error)
            throw retryError
        }
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
            logTag,
            "Blacklisting $backend for text inference after a backend-specific failure on '${model.cacheKey}'",
        )
    }

    private fun buildSafeRetryRequest(request: TextGenerationRequest): TextGenerationRequest? {
        val effectiveUsesVulkan = request.options.useVulkan ?: config.text.useVulkan
        val effectiveUsesFlashAttention =
            request.options.useFlashAttention ?: config.text.useFlashAttention
        val effectiveBatchSize = resolveBlockingBatchSize(config.text, request.batchSize, request.maxTokens)

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
}
