package io.aatricks.llmedge.tools

import java.time.LocalDate

/**
 * Generates system prompts to instruct the LLM on how to format its tool calls.
 */
object ToolPromptGenerator {

    /**
     * Serializes a list of [Tool] definitions into a system prompt string.
     */
    @JvmStatic
    fun generateSystemPrompt(tools: List<Tool>, currentDate: LocalDate = LocalDate.now()): String {
        if (tools.isEmpty()) return "You are a helpful assistant."

        val promptBuilder = StringBuilder()
        promptBuilder.append("You have access to these TOOLS:\n")

        tools.forEach { tool ->
            promptBuilder.append("- ${tool.name}: ${tool.description}. Params: ")
            val params = tool.parameters.map { "${it.key} (${it.value.type})" }
            promptBuilder.append(params.joinToString(", "))
            promptBuilder.append("\n")
        }

        promptBuilder.append("""
            
            RULES:
            1. If you need info from a tool, you MUST call it using this EXACT JSON format:
            {"tool_name": "NAME", "arguments": {"KEY": "VALUE"}}
            2. After calling a tool, you will receive the "TOOL_RESULT".
            3. Once you have the TOOL_RESULT, use it to provide a helpful response to the user in plain text. DO NOT call the tool again with the same arguments.
            4. If you have the final answer, reply in plain text ONLY.

            Current Date: $currentDate.
        """.trimIndent())

        return promptBuilder.toString()
    }
}

