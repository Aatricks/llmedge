package io.aatricks.llmedge.tools

import io.aatricks.llmedge.text.ConversationMessage
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull

enum class ToolKind {
    READ_ONLY,
    ACTION,
}

enum class ToolParameterType {
    STRING,
    NUMBER,
    INTEGER,
    BOOLEAN,
    OBJECT,
    ARRAY,
}

data class ToolParameter(
    val type: ToolParameterType,
    val description: String,
    val required: Boolean = true,
    val enumValues: List<String> = emptyList(),
)

data class ToolSchema(
    val parameters: Map<String, ToolParameter> = emptyMap(),
) {
    internal fun validate(arguments: JsonObject): List<String> {
        val errors = mutableListOf<String>()
        val declaredParameters = parameters.keys

        parameters.forEach { (name, parameter) ->
            val value = arguments[name]
            if (value == null) {
                if (parameter.required) {
                    errors += "Missing required argument '$name'."
                }
                return@forEach
            }

            if (!value.matches(parameter.type)) {
                errors += "Argument '$name' must be ${parameter.type.name.lowercase()}."
                return@forEach
            }

            if (parameter.enumValues.isNotEmpty()) {
                val content = (value as? JsonPrimitive)?.contentOrNull
                if (content == null || content !in parameter.enumValues) {
                    errors +=
                        "Argument '$name' must be one of: ${parameter.enumValues.joinToString(", ")}."
                }
            }
        }

        if (declaredParameters.isEmpty()) {
            return errors
        }

        arguments.keys
            .filterNot(declaredParameters::contains)
            .forEach { name -> errors += "Unexpected argument '$name'." }

        return errors
    }
}

data class ToolResult(
    val text: String,
    val data: JsonObject = buildJsonObject { },
    val isError: Boolean = false,
) {
    companion object {
        @JvmStatic
        fun success(
            text: String,
            data: JsonObject = buildJsonObject { },
        ): ToolResult = ToolResult(text = text, data = data, isError = false)

        @JvmStatic
        fun error(
            text: String,
            data: JsonObject = buildJsonObject { },
        ): ToolResult = ToolResult(text = text, data = data, isError = true)
    }
}

data class Tool(
    val name: String,
    val description: String,
    val kind: ToolKind = ToolKind.READ_ONLY,
    val schema: ToolSchema = ToolSchema(),
    val handler: suspend (JsonObject) -> ToolResult,
)

data class ToolCall(
    val tool: String,
    val arguments: JsonObject = buildJsonObject { },
)

data class ToolCallRequest(
    val tool: Tool,
    val arguments: JsonObject,
    val conversation: List<ConversationMessage>,
    val step: Int,
)

sealed interface ToolDecision {
    data object Allow : ToolDecision

    data class Deny(
        val reason: String,
    ) : ToolDecision
}

fun interface ToolPolicy {
    suspend fun evaluate(request: ToolCallRequest): ToolDecision
}

object ToolPolicies {
    @JvmField
    val DENY_ACTIONS: ToolPolicy =
        ToolPolicy { request ->
            if (request.tool.kind == ToolKind.READ_ONLY) {
                ToolDecision.Allow
            } else {
                ToolDecision.Deny("Action tools require explicit approval.")
            }
        }

    @JvmField
    val ALLOW_ALL: ToolPolicy = ToolPolicy { ToolDecision.Allow }
}

enum class ToolAgentFinishReason {
    COMPLETED,
    MAX_STEPS,
}

data class ToolAgentTraceStep(
    val step: Int,
    val rawModelOutput: String,
    val toolCall: ToolCall? = null,
    val toolResult: ToolResult? = null,
    val toolDeniedReason: String? = null,
)

data class ToolAgentResult(
    val text: String,
    val finishReason: ToolAgentFinishReason,
    val trace: List<ToolAgentTraceStep>,
)

sealed interface ToolAgentEvent {
    data class Started(
        val message: String,
    ) : ToolAgentEvent

    data class ToolCallRequested(
        val call: ToolCall,
    ) : ToolAgentEvent

    data class ToolApproved(
        val call: ToolCall,
    ) : ToolAgentEvent

    data class ToolDenied(
        val call: ToolCall,
        val reason: String,
    ) : ToolAgentEvent

    data class ToolExecuting(
        val call: ToolCall,
    ) : ToolAgentEvent

    data class ToolResultReceived(
        val call: ToolCall,
        val result: ToolResult,
    ) : ToolAgentEvent

    data class TextChunk(
        val value: String,
    ) : ToolAgentEvent

    data class Completed(
        val result: ToolAgentResult,
    ) : ToolAgentEvent

    data class Failed(
        val message: String,
        val cause: Throwable? = null,
    ) : ToolAgentEvent
}

private fun JsonElement.matches(type: ToolParameterType): Boolean =
    when (type) {
        ToolParameterType.STRING -> (this as? JsonPrimitive)?.isString == true
        ToolParameterType.NUMBER -> (this as? JsonPrimitive)?.doubleOrNull != null
        ToolParameterType.INTEGER -> (this as? JsonPrimitive)?.intOrNull != null
        ToolParameterType.BOOLEAN -> (this as? JsonPrimitive)?.booleanOrNull != null
        ToolParameterType.OBJECT -> this is JsonObject
        ToolParameterType.ARRAY -> this is JsonArray
    }
