package io.aatricks.llmedge.text

import io.aatricks.llmedge.model.ModelSpec
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class ConversationSessionSupportTest {
    @Test
    fun `support prepares runtime and commits transcript state`() = runTest {
        val client = mockk<TextClient>()
        val model = mockk<ModelSpec>()
        val runtime = mockk<ManagedTextModel>()
        val options = TextModelOptions()
        val memory = ConversationWindow(maxTurns = 2, maxTokens = 100, stripThinkTags = true)

        every { model.cacheKey } returns "model"
        coEvery { client.prepare(model, options) } returns Unit
        coEvery { client.acquire(model, options) } returns runtime

        val support = ConversationSessionSupport(client, model, options, memory)

        support.prepare()
        support.withRuntime {
            assertEquals(
                listOf(ConversationMessage(ConversationRole.USER, "hello")),
                previewWithUser("hello"),
            )
            commitTurn("hello", "world")
        }

        assertEquals(
            listOf(
                ConversationMessage(ConversationRole.USER, "hello"),
                ConversationMessage(ConversationRole.ASSISTANT, "world"),
            ),
            support.historySnapshot(),
        )
        assertEquals(
            listOf(
                ConversationMessage(ConversationRole.USER, "hello"),
                ConversationMessage(ConversationRole.ASSISTANT, "world"),
                ConversationMessage(ConversationRole.USER, "again"),
            ),
            support.withHistoryPreview("again"),
        )

        coVerify(exactly = 1) { client.prepare(model, options) }
        coVerify(exactly = 1) { client.acquire(model, options) }
    }
}
