package io.aatricks.llmedge.tools

import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.ConversationMessage
import io.aatricks.llmedge.text.ConversationWindow
import io.aatricks.llmedge.text.ConversationSessionSupport
import io.aatricks.llmedge.text.TextClient
import io.aatricks.llmedge.text.TextModelOptions
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow

class ToolAgent internal constructor(
    private val client: TextClient,
    private val tools: List<Tool>,
    private val model: ModelSpec,
    private val memory: ConversationWindow,
    private val systemPrompt: String?,
    private val options: TextModelOptions,
    private val policy: ToolPolicy,
) {
    private val support = ConversationSessionSupport(client, model, options, memory)
    private val invocationExecutor =
        ToolInvocationExecutor(
            tools = tools,
            policy = policy,
            conversationPreview = ::persistentConversationPreview,
        )
    private val toolSystemPrompt by lazy { ToolPromptGenerator.generateSystemPrompt(tools, systemPrompt) }

    suspend fun prepare() {
        support.prepare()
    }

    suspend fun reply(
        message: String,
        maxSteps: Int = 6,
        maxTokens: Int = -1,
        batchSize: Int = 0,
    ): ToolAgentResult =
        support.withRuntime {
            createTurnLoop { prompt, maxTokens, stepBatchSize ->
                complete(prompt, toolSystemPrompt, maxTokens, stepBatchSize)
            }.run(
                message = message,
                maxSteps = maxSteps,
                maxTokens = maxTokens,
                batchSize = batchSize,
            )
        }

    fun stream(
        message: String,
        maxSteps: Int = 6,
        batchSize: Int = 0,
    ): Flow<ToolAgentEvent> =
        flow {
            emit(ToolAgentEvent.Started(message))

            try {
                support.withRuntime {
                    val streamTurnLoop =
                        createTurnLoop { prompt, _, stepBatchSize ->
                            val collector = StreamingToolResponseCollector { emit(ToolAgentEvent.TextChunk(it)) }
                            streamCompletion(prompt, toolSystemPrompt, stepBatchSize).collect { chunk ->
                                collector.append(chunk)
                            }
                            collector.finish()
                        }
                    val result =
                        streamTurnLoop.run(
                            message = message,
                            maxSteps = maxSteps,
                            batchSize = batchSize,
                            callbacks =
                                ToolAgentTurnCallbacks(
                                    onToolCallRequested = { emit(ToolAgentEvent.ToolCallRequested(it)) },
                                    onToolApproved = { emit(ToolAgentEvent.ToolApproved(it)) },
                                    onToolDenied = { call, reason -> emit(ToolAgentEvent.ToolDenied(call, reason)) },
                                    onToolExecuting = { emit(ToolAgentEvent.ToolExecuting(it)) },
                                    onToolResultReceived = { call, result ->
                                        emit(ToolAgentEvent.ToolResultReceived(call, result))
                                    },
                                ),
                        )
                    emit(ToolAgentEvent.Completed(result))
                }
            } catch (t: Throwable) {
                emit(ToolAgentEvent.Failed(t.message ?: "Tool agent failed.", t))
            }
        }

    fun historySnapshot(): List<ConversationMessage> = support.historySnapshot()

    private suspend fun handleToolInvocation(
        message: String,
        step: Int,
        rawModelOutput: String,
        call: ToolCall,
        working: MutableList<ToolPromptMessage>,
        callbacks: ToolAgentTurnCallbacks = ToolAgentTurnCallbacks(),
    ): ToolStepResult =
        invocationExecutor.handle(message, step, rawModelOutput, call, working, callbacks)

    private fun seedWorkingTranscript(message: String): MutableList<ToolPromptMessage> =
        support
            .withHistoryPreview(message)
            .map { promptMessage(it.role, it.content) }
            .toMutableList()

    private fun persistentConversationPreview(message: String): List<ConversationMessage> =
        support.withHistoryPreview(message)

    private fun commitTurn(
        message: String,
        response: String?,
    ) {
        support.commitTurn(message, response)
    }

    private fun renderWorkingPrompt(messages: List<ToolPromptMessage>): String =
        buildString {
            append("Continue the conversation and answer the final user request.\n\n")
            messages.forEach { message ->
                append(message.role.label)
                append(": ")
                append(message.content.trim())
                append('\n')
            }
        }.trimEnd()

    private fun emptyFinalAnswerReminder(): ToolPromptMessage =
        ToolPromptMessage(
            ToolPromptRole.SYSTEM,
            "Your previous response contained no user-visible text after hidden reasoning was removed. " +
                "Do not repeat a tool call you already satisfied. Answer the user now in plain text using the available tool results.",
        )

    private fun createTurnLoop(
        produceResponse: suspend (prompt: String, maxTokens: Int, batchSize: Int) -> String,
    ): ToolAgentTurnLoop =
        ToolAgentTurnLoop(
            seedWorkingTranscript = ::seedWorkingTranscript,
            renderWorkingPrompt = ::renderWorkingPrompt,
            emptyFinalAnswerReminder = ::emptyFinalAnswerReminder,
            commitTurn = ::commitTurn,
            produceResponse = produceResponse,
            handleToolInvocation = ::handleToolInvocation,
        )
}
