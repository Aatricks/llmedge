package io.aatricks.llmedge.text

import org.junit.Assert.assertTrue
import org.junit.Test

class PromptRendererTest {
    @Test
    fun `render includes system prompt and labeled transcript`() {
        val prompt =
            PromptRenderer.render(
                messages =
                    listOf(
                        ConversationMessage(ConversationRole.USER, "Hello"),
                        ConversationMessage(ConversationRole.ASSISTANT, "Hi there"),
                    ),
                systemPrompt = "You are concise.",
            )

        assertTrue(prompt.contains("System: You are concise."))
        assertTrue(prompt.contains("User: Hello"))
        assertTrue(prompt.contains("Assistant: Hi there"))
    }
}
