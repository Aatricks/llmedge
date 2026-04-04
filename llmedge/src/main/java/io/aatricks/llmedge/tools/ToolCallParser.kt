package io.aatricks.llmedge.tools

import io.aatricks.llmedge.text.stripThinkBlocks
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject

internal sealed interface ParsedModelTurn {
    data class ToolInvocation(
        val call: ToolCall,
    ) : ParsedModelTurn

    data class InvalidToolInvocation(
        val reason: String,
    ) : ParsedModelTurn

    data class FinalText(
        val text: String,
    ) : ParsedModelTurn
}

internal object ToolCallParser {
    private val json =
        Json {
            isLenient = true
            ignoreUnknownKeys = true
        }

    fun classify(text: String): ParsedModelTurn {
        val normalized = text.stripThinkBlocks().trim()
        if (normalized.isEmpty()) {
            return ParsedModelTurn.FinalText(normalized)
        }

        extractToolEnvelope(normalized)?.let { candidate ->
            return parseCandidate(candidate)
        }

        return ParsedModelTurn.FinalText(normalized)
    }

    private fun parseCandidate(candidate: String): ParsedModelTurn {
        val parsed = runCatching { json.parseToJsonElement(candidate) }.getOrNull() ?: return ParsedModelTurn.FinalText(candidate)
        val obj = parsed as? JsonObject ?: return ParsedModelTurn.FinalText(candidate)
        val toolKey = when {
            "tool" in obj -> "tool"
            "tool_name" in obj -> "tool_name"
            else -> return ParsedModelTurn.FinalText(candidate)
        }

        val toolName =
            (obj[toolKey] as? JsonPrimitive)?.contentOrNull
                ?: return ParsedModelTurn.InvalidToolInvocation("Tool name must be a string.")

        val argumentsElement = obj["arguments"]
        if (argumentsElement != null && argumentsElement !is JsonObject) {
            return ParsedModelTurn.InvalidToolInvocation("Tool arguments must be a JSON object.")
        }

        return ParsedModelTurn.ToolInvocation(
            ToolCall(
                tool = toolName,
                arguments = argumentsElement?.jsonObject ?: JsonObject(emptyMap()),
            ),
        )
    }

    private fun extractToolEnvelope(text: String): String? {
        if (text.startsWith("{") && text.endsWith("}")) {
            return text
        }

        val fencedMatch =
            """^```(?:json)?\s*(\{.*\})\s*```$"""
                .toRegex(setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE))
                .matchEntire(text)

        return fencedMatch?.groupValues?.get(1)
    }
}
