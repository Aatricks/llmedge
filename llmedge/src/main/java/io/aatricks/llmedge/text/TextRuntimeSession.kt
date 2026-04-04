package io.aatricks.llmedge.text

import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.runExclusive
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow

internal class TextRuntimeSession(
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val updateMetrics: (SmolLM.GenerationMetrics?) -> Unit,
) {
    suspend fun complete(
        runtime: ManagedTextModel,
        prompt: String,
        systemPrompt: String?,
        options: TextModelOptions,
        maxTokens: Int,
        batchSize: Int,
    ): String =
        runtime.runExclusive(scope.inferenceDispatcher) {
            runtime.ensureOpen()
            prepareModel(runtime.model, systemPrompt, options)
            try {
                val effectiveBatchSize = resolveBlockingBatchSize(config.text, batchSize, maxTokens)
                runtime.model.getResponse(prompt, maxTokens, effectiveBatchSize).also {
                    updateMetrics(runtime.model.getLastGenerationMetrics())
                }
            } finally {
                runtime.model.clearKvCache()
            }
        }

    suspend fun chatTurn(
        runtime: ManagedTextModel,
        prompt: String,
        systemPrompt: String?,
        options: TextModelOptions,
        maxTokens: Int,
        batchSize: Int,
        restoreState: ByteArray? = null,
        maxStateBytes: Long,
    ): Pair<String, ByteArray?> =
        runtime.runExclusive(scope.inferenceDispatcher) {
            runtime.ensureOpen()
            if (restoreState != null) {
                runtime.model.setStateBytes(restoreState)
                runtime.model.setThinkingMode(options.thinkingMode)
                options.reasoningBudget?.let(runtime.model::setReasoningBudget)
            } else {
                prepareModel(runtime.model, systemPrompt, options)
            }
            val effectiveBatchSize = resolveBlockingBatchSize(config.text, batchSize, maxTokens)
            val response = runtime.model.getResponse(prompt, maxTokens, effectiveBatchSize)
            updateMetrics(runtime.model.getLastGenerationMetrics())
            val stateBytes = runtime.model.getStateBytes()?.takeIf { it.size <= maxStateBytes }
            runtime.model.clearKvCache()
            response to stateBytes
        }

    fun streamCompletion(
        runtime: ManagedTextModel,
        prompt: String,
        systemPrompt: String?,
        options: TextModelOptions,
        batchSize: Int,
    ): Flow<String> =
        flow {
            runtime.runExclusive {
                runtime.ensureOpen()
                prepareModel(runtime.model, systemPrompt, options)
                try {
                    val effectiveBatchSize = resolveStreamBatchSize(config.text, batchSize)
                    runtime.model
                        .getResponseAsFlow(prompt, scope.inferenceDispatcher, effectiveBatchSize)
                        .buffer(64)
                        .collect { chunk ->
                            if (chunk != "[EOG]") {
                                emit(chunk)
                            }
                        }
                    updateMetrics(runtime.model.getLastGenerationMetrics())
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
}
