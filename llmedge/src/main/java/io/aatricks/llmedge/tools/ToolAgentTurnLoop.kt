package io.aatricks.llmedge.tools

import io.aatricks.llmedge.text.ConversationMessage
import io.aatricks.llmedge.text.ConversationRole
import io.aatricks.llmedge.text.stripThinkBlocks
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

internal class ToolAgentTurnLoop(
    private val seedWorkingTranscript: (String) -> MutableList<ConversationMessage>,
    private val renderWorkingPrompt: (List<ConversationMessage>) -> String,
    private val emptyFinalAnswerReminder: () -> ConversationMessage,
    private val commitTurn: (String, String?) -> Unit,
    private val produceResponse: suspend (prompt: String, maxTokens: Int, batchSize: Int) -> String,
    private val handleToolInvocation: suspend (
        message: String,
        step: Int,
        rawModelOutput: String,
        call: ToolCall,
        working: MutableList<ConversationMessage>,
        callbacks: ToolAgentTurnCallbacks,
    ) -> ToolStepResult,
    private val conversationWindow: io.aatricks.llmedge.text.ConversationWindow? = null,
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
            conversationWindow?.let { window ->
                val trimmed = window.trim(working)
                working.clear()
                working.addAll(trimmed)
            }
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
                    working += ConversationMessage(ConversationRole.ASSISTANT, response)
                    working += ConversationMessage(ConversationRole.TOOL, formatToolResultMessage("invalid_tool_call", result))
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

internal fun formatToolResultMessage(
    toolName: String,
    result: ToolResult,
): String =
    buildString {
        append("Tool '")
        append(toolName)
        append("' returned this JSON result:\n")
        
        val cap = 8000
        val cappedResult = if (result.text.length > cap) {
            result.copy(text = result.text.take(cap) + "\n... [TRUNCATED]")
        } else {
            result
        }
        
        append(
            Json.encodeToString(
                JsonObject.serializer(),
                buildJsonObject {
                    put("tool", toolName)
                    put("ok", !cappedResult.isError)
                    put("text", cappedResult.text)
                    put("data", cappedResult.data)
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
