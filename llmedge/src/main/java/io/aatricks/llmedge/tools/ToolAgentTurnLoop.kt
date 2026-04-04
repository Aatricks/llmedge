package io.aatricks.llmedge.tools

import io.aatricks.llmedge.text.ConversationRole
import io.aatricks.llmedge.text.stripThinkBlocks
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

internal class ToolAgentTurnLoop(
    private val seedWorkingTranscript: (String) -> MutableList<ToolPromptMessage>,
    private val renderWorkingPrompt: (List<ToolPromptMessage>) -> String,
    private val emptyFinalAnswerReminder: () -> ToolPromptMessage,
    private val commitTurn: (String, String?) -> Unit,
    private val produceResponse: suspend (prompt: String, maxTokens: Int, batchSize: Int) -> String,
    private val handleToolInvocation: suspend (
        message: String,
        step: Int,
        rawModelOutput: String,
        call: ToolCall,
        working: MutableList<ToolPromptMessage>,
        callbacks: ToolAgentTurnCallbacks,
    ) -> ToolStepResult,
) {
    suspend fun run(
        message: String,
        maxSteps: Int,
        maxTokens: Int = -1,
        batchSize: Int = 0,
        callbacks: ToolAgentTurnCallbacks = ToolAgentTurnCallbacks(),
    ): ToolAgentResult {
        require(maxSteps > 0) { "maxSteps must be greater than 0." }

        val working = seedWorkingTranscript(message)
        val trace = mutableListOf<ToolAgentTraceStep>()
        var finalText = ""

        for (step in 1..maxSteps) {
            val response = produceResponse(renderWorkingPrompt(working), maxTokens, batchSize)

            when (val turn = ToolCallParser.classify(response)) {
                is ParsedModelTurn.FinalText -> {
                    trace += ToolAgentTraceStep(step = step, rawModelOutput = response)
                    val visibleText = turn.text.stripThinkBlocks()
                    if (visibleText.isNotBlank()) {
                        finalText = visibleText
                        callbacks.onTextChunk(finalText)
                        commitTurn(message, finalText)
                        return ToolAgentResult(
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
                    callbacks.onToolResultReceived(ToolCall("invalid_tool_call"), result)
                }

                is ParsedModelTurn.ToolInvocation -> {
                    callbacks.onToolCallRequested(turn.call)
                    val stepResult =
                        handleToolInvocation(
                            message,
                            step,
                            response,
                            turn.call,
                            working,
                            callbacks,
                        )
                    trace += stepResult.trace
                }
            }
        }

        return ToolAgentResult(
            text = finalText,
            finishReason = ToolAgentFinishReason.MAX_STEPS,
            trace = trace.toList(),
        ).also { result ->
            commitTurn(message, finalText.takeUnless(String::isBlank))
            callbacks.onCompleted(result)
        }
    }
}

internal data class ToolAgentTurnCallbacks(
    val onToolCallRequested: suspend (ToolCall) -> Unit = {},
    val onToolApproved: suspend (ToolCall) -> Unit = {},
    val onToolDenied: suspend (ToolCall, String) -> Unit = { _, _ -> },
    val onToolExecuting: suspend (ToolCall) -> Unit = {},
    val onToolResultReceived: suspend (ToolCall, ToolResult) -> Unit = { _, _ -> },
    val onTextChunk: suspend (String) -> Unit = {},
    val onCompleted: suspend (ToolAgentResult) -> Unit = {},
)

internal data class ToolStepResult(
    val trace: ToolAgentTraceStep,
)

internal enum class ToolPromptRole {
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL,
}

internal data class ToolPromptMessage(
    val role: ToolPromptRole,
    val content: String,
)

internal val ToolPromptRole.label: String
    get() =
        when (this) {
            ToolPromptRole.SYSTEM -> "System"
            ToolPromptRole.USER -> "User"
            ToolPromptRole.ASSISTANT -> "Assistant"
            ToolPromptRole.TOOL -> "Tool"
        }

internal fun promptMessage(
    role: ConversationRole,
    content: String,
): ToolPromptMessage =
    when (role) {
        ConversationRole.SYSTEM -> ToolPromptMessage(ToolPromptRole.SYSTEM, content)
        ConversationRole.USER -> ToolPromptMessage(ToolPromptRole.USER, content)
        ConversationRole.ASSISTANT -> ToolPromptMessage(ToolPromptRole.ASSISTANT, content)
    }

internal fun formatToolResultMessage(
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

internal fun toolErrorData(
    code: String,
    message: String,
): JsonObject =
    buildJsonObject {
        put("code", code)
        put("message", message)
    }
