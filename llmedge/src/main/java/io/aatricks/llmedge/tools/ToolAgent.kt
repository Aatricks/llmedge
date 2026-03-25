package io.aatricks.llmedge.tools

import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.ConversationMessage
import io.aatricks.llmedge.text.ConversationRole
import io.aatricks.llmedge.text.ConversationWindow
import io.aatricks.llmedge.text.TextClient
import io.aatricks.llmedge.text.TextModelOptions
import io.aatricks.llmedge.text.stripThinkBlocks
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

class ToolAgent internal constructor(
    private val client: TextClient,
    private val tools: List<Tool>,
    private val model: ModelSpec,
    private val memory: ConversationWindow,
    private val systemPrompt: String?,
    private val options: TextModelOptions,
    private val policy: ToolPolicy,
) {
    private val sessionMutex = Mutex()
    private val history = mutableListOf<ConversationMessage>()
    private val toolsByName = tools.associateBy(Tool::name)
    private val toolSystemPrompt by lazy { ToolPromptGenerator.generateSystemPrompt(tools, systemPrompt) }

    suspend fun prepare() {
        client.prepare(model, options)
    }

    suspend fun reply(
        message: String,
        maxSteps: Int = 6,
        maxTokens: Int = -1,
        batchSize: Int = 0,
    ): ToolAgentResult {
        require(maxSteps > 0) { "maxSteps must be greater than 0." }

        return sessionMutex.withLock {
            val runtime = client.acquire(model, options)
            val working = seedWorkingTranscript(message)
            val trace = mutableListOf<ToolAgentTraceStep>()
            var finalText = ""

            for (step in 1..maxSteps) {
                val response =
                    client.complete(
                        runtime = runtime,
                        prompt = renderWorkingPrompt(working),
                        systemPrompt = toolSystemPrompt,
                        options = options,
                        maxTokens = maxTokens,
                        batchSize = batchSize,
                    )

                when (val turn = ToolCallParser.classify(response)) {
                    is ParsedModelTurn.FinalText -> {
                        trace += ToolAgentTraceStep(step = step, rawModelOutput = response)
                        val visibleText = turn.text.stripThinkBlocks()
                        if (visibleText.isNotBlank()) {
                            finalText = visibleText
                            commitTurn(message, finalText)
                            return@withLock ToolAgentResult(
                                text = finalText,
                                finishReason = ToolAgentFinishReason.COMPLETED,
                                trace = trace.toList(),
                            )
                        }

                        working += emptyFinalAnswerReminder()
                    }

                    is ParsedModelTurn.InvalidToolInvocation -> {
                        val result = ToolResult.error(turn.reason, toolErrorData("invalid_tool_call", turn.reason))
                        trace +=
                            ToolAgentTraceStep(
                                step = step,
                                rawModelOutput = response,
                                toolResult = result,
                            )
                        working += ToolPromptMessage(ToolPromptRole.ASSISTANT, response)
                        working += ToolPromptMessage(ToolPromptRole.TOOL, formatToolResultMessage("invalid_tool_call", result))
                    }

                    is ParsedModelTurn.ToolInvocation -> {
                        val stepResult = handleToolInvocation(message, step, response, turn.call, working)
                        trace += stepResult.trace
                    }
                }
            }

            val result =
                ToolAgentResult(
                    text = finalText,
                    finishReason = ToolAgentFinishReason.MAX_STEPS,
                    trace = trace.toList(),
                )
            commitTurn(message, finalText.takeUnless(String::isBlank))
            result
        }
    }

    fun stream(
        message: String,
        maxSteps: Int = 6,
        batchSize: Int = 0,
    ): Flow<ToolAgentEvent> =
        flow {
            require(maxSteps > 0) { "maxSteps must be greater than 0." }
            emit(ToolAgentEvent.Started(message))

            try {
                sessionMutex.withLock {
                    val runtime = client.acquire(model, options)
                    val working = seedWorkingTranscript(message)
                    val trace = mutableListOf<ToolAgentTraceStep>()
                    var finalText = ""

                    for (step in 1..maxSteps) {
                        val prompt = renderWorkingPrompt(working)
                        val chunks = mutableListOf<String>()

                        client
                            .streamCompletion(
                                runtime = runtime,
                                prompt = prompt,
                                systemPrompt = toolSystemPrompt,
                                options = options,
                                batchSize = batchSize,
                            ).collect { chunk -> chunks += chunk }

                        val response = chunks.joinToString("")

                        when (val turn = ToolCallParser.classify(response)) {
                            is ParsedModelTurn.FinalText -> {
                                trace += ToolAgentTraceStep(step = step, rawModelOutput = response)
                                val visibleText = turn.text.stripThinkBlocks()
                                if (visibleText.isNotBlank()) {
                                    finalText = visibleText
                                    emit(ToolAgentEvent.TextChunk(finalText))
                                    commitTurn(message, finalText)
                                    emit(
                                        ToolAgentEvent.Completed(
                                            ToolAgentResult(
                                                text = finalText,
                                                finishReason = ToolAgentFinishReason.COMPLETED,
                                                trace = trace.toList(),
                                            ),
                                        ),
                                    )
                                    return@withLock
                                }

                                working += emptyFinalAnswerReminder()
                            }

                            is ParsedModelTurn.InvalidToolInvocation -> {
                                val result = ToolResult.error(turn.reason, toolErrorData("invalid_tool_call", turn.reason))
                                trace +=
                                    ToolAgentTraceStep(
                                        step = step,
                                        rawModelOutput = response,
                                        toolResult = result,
                                    )
                                working += ToolPromptMessage(ToolPromptRole.ASSISTANT, response)
                                working += ToolPromptMessage(ToolPromptRole.TOOL, formatToolResultMessage("invalid_tool_call", result))
                                emit(
                                    ToolAgentEvent.ToolResultReceived(
                                        ToolCall("invalid_tool_call"),
                                        result,
                                    ),
                                )
                            }

                            is ParsedModelTurn.ToolInvocation -> {
                                emit(ToolAgentEvent.ToolCallRequested(turn.call))
                                val stepResult =
                                    handleToolInvocation(
                                        message = message,
                                        step = step,
                                        rawModelOutput = response,
                                        call = turn.call,
                                        working = working,
                                        onApproved = { emit(ToolAgentEvent.ToolApproved(it)) },
                                        onDenied = { callValue, reason -> emit(ToolAgentEvent.ToolDenied(callValue, reason)) },
                                        onExecuting = { emit(ToolAgentEvent.ToolExecuting(it)) },
                                        onResult = { callValue, result -> emit(ToolAgentEvent.ToolResultReceived(callValue, result)) },
                                    )
                                trace += stepResult.trace
                            }
                        }
                    }

                    val result =
                        ToolAgentResult(
                            text = finalText,
                            finishReason = ToolAgentFinishReason.MAX_STEPS,
                            trace = trace.toList(),
                        )
                    commitTurn(message, finalText.takeUnless(String::isBlank))
                    emit(ToolAgentEvent.Completed(result))
                }
            } catch (t: Throwable) {
                emit(ToolAgentEvent.Failed(t.message ?: "Tool agent failed.", t))
            }
        }

    fun historySnapshot(): List<ConversationMessage> = history.toList()

    private suspend fun handleToolInvocation(
        message: String,
        step: Int,
        rawModelOutput: String,
        call: ToolCall,
        working: MutableList<ToolPromptMessage>,
        onApproved: suspend (ToolCall) -> Unit = {},
        onDenied: suspend (ToolCall, String) -> Unit = { _, _ -> },
        onExecuting: suspend (ToolCall) -> Unit = {},
        onResult: suspend (ToolCall, ToolResult) -> Unit = { _, _ -> },
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
            onResult(call, result)
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
            onResult(call, result)
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
                ToolDecision.Allow -> onApproved(call)
                is ToolDecision.Deny -> {
                    val result = ToolResult.error(decision.reason, toolErrorData("action_denied", decision.reason))
                    working += ToolPromptMessage(ToolPromptRole.TOOL, formatToolResultMessage(call.tool, result))
                    onDenied(call, decision.reason)
                    onResult(call, result)
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

        onExecuting(call)
        val result =
            runCatching { tool.handler(call.arguments) }
                .getOrElse { error ->
                    ToolResult.error(
                        "Error executing tool '${tool.name}': ${error.message ?: "Unknown error."}",
                        toolErrorData("execution_failed", error.message ?: "Unknown error."),
                    )
                }

        working += ToolPromptMessage(ToolPromptRole.TOOL, formatToolResultMessage(call.tool, result))
        onResult(call, result)
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
        memory
            .trim(history + ConversationMessage(ConversationRole.USER, message))
            .map { promptMessage(it.role, it.content) }
            .toMutableList()

    private fun persistentConversationPreview(message: String): List<ConversationMessage> =
        memory.trim(history + ConversationMessage(ConversationRole.USER, message))

    private fun commitTurn(
        message: String,
        response: String?,
    ) {
        history += ConversationMessage(ConversationRole.USER, message)
        response
            ?.takeUnless(String::isBlank)
            ?.let { history += ConversationMessage(ConversationRole.ASSISTANT, it) }
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

    private fun promptMessage(
        role: ConversationRole,
        content: String,
    ): ToolPromptMessage =
        when (role) {
            ConversationRole.SYSTEM -> ToolPromptMessage(ToolPromptRole.SYSTEM, content)
            ConversationRole.USER -> ToolPromptMessage(ToolPromptRole.USER, content)
            ConversationRole.ASSISTANT -> ToolPromptMessage(ToolPromptRole.ASSISTANT, content)
        }

    private fun formatToolResultMessage(
        toolName: String,
        result: ToolResult,
    ): String =
        buildString {
            append("Tool '")
            append(toolName)
            append("' returned this JSON result:\n")
            append(
                Json.encodeToString(
                    JsonObject.serializer(),
                    buildJsonObject {
                        put("tool", toolName)
                        put("ok", !result.isError)
                        put("text", result.text)
                        put("data", result.data)
                    },
                ),
            )
            append("\nUse it to continue.")
        }

    private fun toolErrorData(
        code: String,
        message: String,
    ): JsonObject =
        buildJsonObject {
            put("code", code)
            put("message", message)
        }

    private fun emptyFinalAnswerReminder(): ToolPromptMessage =
        ToolPromptMessage(
            ToolPromptRole.SYSTEM,
            "Your previous response contained no user-visible text after hidden reasoning was removed. " +
                "Do not repeat a tool call you already satisfied. Answer the user now in plain text using the available tool results.",
        )
}

private data class ToolStepResult(
    val trace: ToolAgentTraceStep,
)

private enum class ToolPromptRole {
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL,
}

private data class ToolPromptMessage(
    val role: ToolPromptRole,
    val content: String,
)

private val ToolPromptRole.label: String
    get() =
        when (this) {
            ToolPromptRole.SYSTEM -> "System"
            ToolPromptRole.USER -> "User"
            ToolPromptRole.ASSISTANT -> "Assistant"
            ToolPromptRole.TOOL -> "Tool"
        }
