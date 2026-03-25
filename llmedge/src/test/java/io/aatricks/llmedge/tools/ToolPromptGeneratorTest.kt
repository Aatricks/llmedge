package io.aatricks.llmedge.tools

import java.time.LocalDate
import org.junit.Assert.assertTrue
import org.junit.Test

class ToolPromptGeneratorTest {
    @Test
    fun `generateSystemPrompt includes structured tool schema and date`() {
        val prompt =
            ToolPromptGenerator.generateSystemPrompt(
                tools =
                    listOf(
                        Tool(
                            name = "search",
                            description = "Searches docs",
                            schema =
                                ToolSchema(
                                    parameters =
                                        mapOf(
                                            "query" to
                                                ToolParameter(
                                                    type = ToolParameterType.STRING,
                                                    description = "search query",
                                                ),
                                        ),
                            ),
                            handler = { ToolResult.success("ok") },
                        ),
                        Tool(
                            name = "clock",
                            description = "Returns the current time",
                            handler = { ToolResult.success("ok") },
                        ),
                    ),
                baseSystemPrompt = "You are concise.",
                currentDate = LocalDate.of(2026, 3, 8),
            )

        assertTrue(prompt.contains("You are concise."))
        assertTrue(prompt.contains("search [read_only]: Searches docs"))
        assertTrue(prompt.contains("clock [read_only]: Returns the current time Params: none"))
        assertTrue(prompt.contains("""{"tool":"tool_name","arguments":{"arg":"value"}}"""))
        assertTrue(prompt.contains("""{"tool":"tool_name","arguments":{}}"""))
        assertTrue(prompt.contains("query (string)"))
        assertTrue(prompt.contains("Current Date: 2026-03-08."))
    }
}
