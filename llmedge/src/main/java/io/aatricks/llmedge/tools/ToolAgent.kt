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
    private val toolsByName = tools.associateBy(Tool::name)
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
                            val chunks = mutableListOf<String>()
                            streamCompletion(prompt, toolSystemPrompt, stepBatchSize).collect { chunk ->
                                chunks += chunk
                            }
                            chunks.joinToString("")
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
                                    onTextChunk = { emit(ToolAgentEvent.TextChunk(it)) },
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
    ): ToolStepResult {
        working += ToolPromptMessage(ToolPromptRole.ASSISTANT, rawModelOutput)
        val tool = toolsByName[call.tool]

        if (tool == null) {
            val result =
                ToolResult.error(
                    "Tool '${call.tool}' is not registered.",
                    toolErrorData("unknown_tool", "Tool '${call.tool}' is not registered."),
                )
            working += ToolPromptMessage(ToolPromptRole.TOOL, formatToolResultMessage(call.tool, result))
            callbacks.onToolResultReceived(call, result)
            return ToolStepResult(
                ToolAgentTraceStep(
                    step = step,
                    rawModelOutput = rawModelOutput,
                    toolCall = call,
                    toolResult = result,
                ),
            )
        }

        val validationErrors = tool.schema.validate(call.arguments)
        if (validationErrors.isNotEmpty()) {
            val result =
                ToolResult.error(
                    validationErrors.joinToString(" "),
                    toolErrorData("invalid_arguments", validationErrors.joinToString(" ")),
                )
            working += ToolPromptMessage(ToolPromptRole.TOOL, formatToolResultMessage(call.tool, result))
            callbacks.onToolResultReceived(call, result)
            return ToolStepResult(
                ToolAgentTraceStep(
                    step = step,
                    rawModelOutput = rawModelOutput,
                    toolCall = call,
                    toolResult = result,
                ),
            )
        }

        if (tool.kind == ToolKind.ACTION) {
            when (val decision = policy.evaluate(ToolCallRequest(tool, call.arguments, persistentConversationPreview(message), step))) {
                ToolDecision.Allow -> callbacks.onToolApproved(call)
                is ToolDecision.Deny -> {
                    val result = ToolResult.error(decision.reason, toolErrorData("action_denied", decision.reason))
                    working += ToolPromptMessage(ToolPromptRole.TOOL, formatToolResultMessage(call.tool, result))
                    callbacks.onToolDenied(call, decision.reason)
                    callbacks.onToolResultReceived(call, result)
                    return ToolStepResult(
                        ToolAgentTraceStep(
                            step = step,
                            rawModelOutput = rawModelOutput,
                            toolCall = call,
                            toolResult = result,
                            toolDeniedReason = decision.reason,
                        ),
                    )
                }
            }
        }

        callbacks.onToolExecuting(call)
        val result =
            runCatching { tool.handler(call.arguments) }
                .getOrElse { error ->
                    ToolResult.error(
                        "Error executing tool '${tool.name}': ${error.message ?: "Unknown error."}",
                        toolErrorData("execution_failed", error.message ?: "Unknown error."),
                    )
                }

        working += ToolPromptMessage(ToolPromptRole.TOOL, formatToolResultMessage(call.tool, result))
        callbacks.onToolResultReceived(call, result)
        return ToolStepResult(
            ToolAgentTraceStep(
                step = step,
                rawModelOutput = rawModelOutput,
                toolCall = call,
                toolResult = result,
            ),
        )
    }

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
