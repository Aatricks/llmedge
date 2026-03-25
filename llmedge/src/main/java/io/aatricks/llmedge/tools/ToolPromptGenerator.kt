package io.aatricks.llmedge.tools

import java.time.LocalDate

/**
 * Generates system prompts to instruct the LLM on how to format its tool calls.
 */
object ToolPromptGenerator {
    @JvmStatic
    fun generateSystemPrompt(
        tools: List<Tool>,
        baseSystemPrompt: String? = null,
        currentDate: LocalDate = LocalDate.now(),
    ): String {
        val promptBuilder = StringBuilder()

        baseSystemPrompt
            ?.takeUnless(String::isBlank)
            ?.let {
                promptBuilder.append(it.trim())
                promptBuilder.append("\n\n")
            }

        if (tools.isEmpty()) {
            promptBuilder.append("You are a helpful assistant.")
            return promptBuilder.toString()
        }

        promptBuilder.append("You can use these tools when necessary.\n")
        tools.forEach { tool ->
            promptBuilder.append("- ")
            promptBuilder.append(tool.name)
            promptBuilder.append(" [")
            promptBuilder.append(tool.kind.name.lowercase())
            promptBuilder.append("]: ")
            promptBuilder.append(tool.description)
            if (tool.schema.parameters.isNotEmpty()) {
                promptBuilder.append(" Params: ")
                promptBuilder.append(
                    tool.schema.parameters.entries.joinToString(", ") { (name, parameter) ->
                        buildString {
                            append(name)
                            append(" (")
                            append(parameter.type.name.lowercase())
                            if (!parameter.required) {
                                append(", optional")
                            }
                            if (parameter.enumValues.isNotEmpty()) {
                                append(", enum=")
                                append(parameter.enumValues.joinToString("|"))
                            }
                            append(")")
                        }
                    },
                )
            } else {
                promptBuilder.append(" Params: none (send an empty arguments object).")
            }
            promptBuilder.append('\n')
        }

        promptBuilder.append(
            """
            |
            |Rules:
            |1. If a tool is needed, reply with JSON only using this exact shape:
            |{"tool":"tool_name","arguments":{"arg":"value"}}
            |2. For tools with no parameters, send an empty arguments object:
            |{"tool":"tool_name","arguments":{}}
            |3. Do not wrap the JSON in prose when calling a tool.
            |4. For live device state like time, battery, or device info, do not guess. Call the relevant tool and use its result.
            |5. After a tool result appears, use it to continue. If the tool result reports an error, either fix the call or answer without that tool.
            |6. Copy numeric values, booleans, and URLs from tool results exactly. Do not change or invent them.
            |7. Only claim an action succeeded if the tool result for that action says it succeeded.
            |8. Do not repeat the same tool call with the same arguments if you already have a usable result. Once you have enough information, answer in plain text.
            |9. If no tool is needed, reply in plain text only.
            |
            |Current Date: $currentDate.
            """.trimMargin(),
        )

        return promptBuilder.toString().trim()
    }
}
