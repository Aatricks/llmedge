package io.aatricks.llmedge.text

import org.junit.Assert.assertEquals
import org.junit.Test

class SessionTranscriptTest {
    @Test
    fun `preview includes pending user message without mutating history`() {
        val transcript = SessionTranscript(ConversationWindow(maxTurns = 2, maxTokens = 100))

        val preview = transcript.previewWithUser("hello")

        assertEquals(listOf(ConversationMessage(ConversationRole.USER, "hello")), preview)
        assertEquals(emptyList<ConversationMessage>(), transcript.snapshot())
    }

    @Test
    fun `commit turn appends user and non blank assistant response`() {
        val transcript = SessionTranscript(ConversationWindow(maxTurns = 2, maxTokens = 100))

        transcript.commitTurn("hello", "world")

        assertEquals(
            listOf(
                ConversationMessage(ConversationRole.USER, "hello"),
                ConversationMessage(ConversationRole.ASSISTANT, "world"),
            ),
            transcript.snapshot(),
        )
    }

    @Test
    fun `commit turn preserves blank assistant response`() {
        val transcript = SessionTranscript(ConversationWindow(maxTurns = 2, maxTokens = 100))

        transcript.commitTurn("hello", "   ")

        assertEquals(
            listOf(
                ConversationMessage(ConversationRole.USER, "hello"),
                ConversationMessage(ConversationRole.ASSISTANT, "   "),
            ),
            transcript.snapshot(),
        )
    }

    @Test
    fun `commit turn ignores null assistant response`() {
        val transcript = SessionTranscript(ConversationWindow(maxTurns = 2, maxTokens = 100))

        transcript.commitTurn("hello", null)

        assertEquals(
            listOf(ConversationMessage(ConversationRole.USER, "hello")),
            transcript.snapshot(),
        )
    }
}
