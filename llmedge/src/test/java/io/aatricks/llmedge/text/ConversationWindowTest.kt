package io.aatricks.llmedge.text

import org.junit.Assert.assertEquals
import org.junit.Test

class ConversationWindowTest {
    @Test
    fun `trim strips think tags and keeps latest turns`() {
        val window = ConversationWindow(maxTurns = 1, maxTokens = 100, stripThinkTags = true)

        val trimmed =
            window.trim(
                listOf(
                    ConversationMessage(ConversationRole.USER, "Question 1"),
                    ConversationMessage(ConversationRole.ASSISTANT, "<think>private</think> Visible answer"),
                    ConversationMessage(ConversationRole.USER, "Question 2"),
                ),
            )

        assertEquals(
            listOf(
                ConversationMessage(ConversationRole.ASSISTANT, "Visible answer"),
                ConversationMessage(ConversationRole.USER, "Question 2"),
            ),
            trimmed,
        )
    }

    @Test
    fun `trim drops oldest messages to fit token budget`() {
        val window = ConversationWindow(maxTurns = 3, maxTokens = 5, stripThinkTags = false)

        val trimmed =
            window.trim(
                listOf(
                    ConversationMessage(ConversationRole.USER, "1234567890"),
                    ConversationMessage(ConversationRole.ASSISTANT, "1234567890"),
                    ConversationMessage(ConversationRole.USER, "ok"),
                ),
            )

        assertEquals(listOf(ConversationMessage(ConversationRole.USER, "ok")), trimmed)
    }
}
