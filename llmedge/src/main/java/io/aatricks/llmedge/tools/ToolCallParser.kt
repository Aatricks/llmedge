package io.aatricks.llmedge.tools

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
        var invalidReason: String? = null

        extractCandidates(text).forEach { candidate ->
            when (val parsed = parseCandidate(candidate)) {
                is ParsedModelTurn.ToolInvocation -> return parsed
                is ParsedModelTurn.InvalidToolInvocation -> if (invalidReason == null) invalidReason = parsed.reason
                is ParsedModelTurn.FinalText -> Unit
            }
        }

        return invalidReason?.let(ParsedModelTurn::InvalidToolInvocation)
            ?: ParsedModelTurn.FinalText(text.trim())
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

    private fun extractCandidates(text: String): List<String> {
        val fenced =
            """```(?:json)?\s*(\{.*?\})\s*```"""
                .toRegex(setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE))
                .findAll(text)
                .map { it.groupValues[1] }
                .toList()

        val balancedObjects = mutableListOf<String>()
        var start = -1
        var depth = 0
        var inString = false
        var escaped = false

        text.forEachIndexed { index, char ->
            if (escaped) {
                escaped = false
                return@forEachIndexed
            }

            when (char) {
                '\\' -> if (inString) escaped = true
                '"' -> inString = !inString
                '{' -> if (!inString) {
                    if (depth == 0) {
                        start = index
                    }
                    depth++
                }

                '}' -> if (!inString && depth > 0) {
                    depth--
                    if (depth == 0 && start >= 0) {
                        balancedObjects += text.substring(start, index + 1)
                        start = -1
                    }
                }
            }
        }

        return buildList {
            addAll(fenced)
            addAll(balancedObjects)
            add(text.trim())
        }.distinct()
    }
}
