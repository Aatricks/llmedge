package io.aatricks.llmedge.tools

import java.time.LocalDate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ToolPromptGeneratorTest {
    @Test
    fun `generateSystemPrompt includes tools and provided date`() {
        val prompt =
            ToolPromptGenerator.generateSystemPrompt(
                tools = listOf(
                    Tool(
                        name = "search",
                        description = "Searches docs",
                        parameters = mapOf("query" to ParameterDescription("string", "search query")),
                        execute = { "ok" },
                    ),
                ),
                currentDate = LocalDate.of(2026, 3, 8),
            )

        assertTrue(prompt.contains("search: Searches docs"))
        assertTrue(prompt.contains("query (string)"))
        assertTrue(prompt.contains("Current Date: 2026-03-08."))
    }

    @Test
    fun `generateSystemPrompt returns friendly default without tools`() {
        assertEquals("You are a helpful assistant.", ToolPromptGenerator.generateSystemPrompt(emptyList()))
    }
}