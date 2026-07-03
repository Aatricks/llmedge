package io.aatricks.llmedge.tools

import io.aatricks.llmedge.text.ConversationMessage
import io.aatricks.llmedge.text.ConversationRole
import kotlinx.coroutines.CancellationException

internal class ToolInvocationExecutor(
    tools: List<Tool>,
    private val policy: ToolPolicy,
    private val conversationPreview: (String) -> List<ConversationMessage>,
) {
    private val toolsByName: Map<String, Tool>

    init {
        val names = tools.map { it.name }
        if (names.size != names.distinct().size) {
            val duplicates = names.groupBy { it }.filter { it.value.size > 1 }.keys
            throw IllegalArgumentException("Duplicate tool names registered: $duplicates")
        }
        toolsByName = tools.associateBy(Tool::name)
    }

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
        val result = try {
            tool.handler(call.arguments)
        } catch (e: CancellationException) {
            throw e
        } catch (error: Throwable) {
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
