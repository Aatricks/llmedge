package io.aatricks.llmedge.tools

import io.aatricks.llmedge.text.ConversationMessage
import io.aatricks.llmedge.text.ConversationRole

internal class ToolInvocationExecutor(
    tools: List<Tool>,
    private val policy: ToolPolicy,
    private val conversationPreview: (String) -> List<ConversationMessage>,
) {
    private val toolsByName = tools.associateBy(Tool::name)

    suspend fun handle(
        message: String,
        step: Int,
        rawModelOutput: String,
        call: ToolCall,
        working: MutableList<ConversationMessage>,
        callbacks: ToolAgentTurnCallbacks = ToolAgentTurnCallbacks(),
    ): ToolStepResult {
        working += ConversationMessage(ConversationRole.ASSISTANT, rawModelOutput)
        val tool = toolsByName[call.tool]

        if (tool == null) {
            val result =
                ToolResult.error(
                    "Tool '${call.tool}' is not registered.",
                    toolErrorData("unknown_tool", "Tool '${call.tool}' is not registered."),
                )
            working += ConversationMessage(ConversationRole.TOOL, formatToolResultMessage(call.tool, result))
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
            working += ConversationMessage(ConversationRole.TOOL, formatToolResultMessage(call.tool, result))
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
            when (
                val decision =
                    policy.evaluate(
                        ToolCallRequest(
                            tool,
                            call.arguments,
                            conversationPreview(message),
                            step,
                        ),
                    )
            ) {
                ToolDecision.Allow -> callbacks.onToolApproved(call)
                is ToolDecision.Deny -> {
                    val result =
                        ToolResult.error(
                            decision.reason,
                            toolErrorData("action_denied", decision.reason),
                        )
                    working += ConversationMessage(ConversationRole.TOOL, formatToolResultMessage(call.tool, result))
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

        working += ConversationMessage(ConversationRole.TOOL, formatToolResultMessage(call.tool, result))
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
}
