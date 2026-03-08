package io.aatricks.llmedge

import java.io.File
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Linux-host end-to-end tests for [ChatSession] using a real native SmolLM library and GGUF model.
 *
 * The assertions focus on deterministic properties that must hold for the user-facing API on a real
 * model: non-empty completions, persisted history, system prompt injection, sliding-window replay,
 * and `<think>` stripping effects on native context usage.
 */
@RunWith(RobolectricTestRunner::class)
@Config(manifest = Config.NONE, sdk = [34])
class ChatSessionLinuxE2ETest {
    @Test
    fun `desktop end-to-end chat session injects system prompt and stores sendMessage history`() = runBlocking {
        val modelPath = resolveModelPath()

        val withoutSystem = runFreshTurn(modelPath, systemPrompt = null, prompt = NATURAL_PROMPT)
        val withSystem = runFreshTurn(modelPath, systemPrompt = LONG_SYSTEM_PROMPT, prompt = NATURAL_PROMPT)

        println("[ChatSessionLinuxE2ETest] withoutSystemContext=${withoutSystem.contextAfterTurn}")
        println("[ChatSessionLinuxE2ETest] withSystemContext=${withSystem.contextAfterTurn}")

        assertTrue("sendMessage should return a non-empty response", withoutSystem.response.isNotBlank())
        assertTrue("sendMessage with system prompt should return a non-empty response", withSystem.response.isNotBlank())
        assertEquals(
            listOf(Role.USER, Role.ASSISTANT),
            withSystem.history.map(ChatMessage::role),
        )
        assertTrue(
            "Injecting a long system prompt should increase native context usage. without=${withoutSystem.contextAfterTurn}, with=${withSystem.contextAfterTurn}",
            withSystem.contextAfterTurn > withoutSystem.contextAfterTurn,
        )
    }

    @Test
    fun `desktop end-to-end chat session streaming persists reconstructed reply`() = runBlocking {
        val modelPath = resolveModelPath()
        val streamedRun = runStreamedTurn(modelPath, prompt = NATURAL_PROMPT)

        println("[ChatSessionLinuxE2ETest] streamedResponse=${streamedRun.response}")

        assertTrue("sendMessageStream should emit a non-empty response", streamedRun.response.isNotBlank())
        assertEquals(
            listOf(Role.USER, Role.ASSISTANT),
            streamedRun.history.map(ChatMessage::role),
        )
        assertEquals(
            "The collected stream should be persisted verbatim in ChatSession history.",
            streamedRun.response,
            streamedRun.history.last().content,
        )
        assertTrue("Streaming should populate native context", streamedRun.contextAfterTurn > 0)
    }

    @Test
    fun `desktop end-to-end chat session sliding window limits replayed context`() = runBlocking {
        val modelPath = resolveModelPath()
        val seededHistory = buildSeededHistory(messageCount = 6)

        val smallWindow =
            runSeededPartialTurn(
                modelPath = modelPath,
                seededHistory = seededHistory,
                maxHistoryMessages = 2,
                stripReasoningFromHistory = false,
                prompt = SHORT_REPLY_PROMPT,
            )
        val largeWindow =
            runSeededPartialTurn(
                modelPath = modelPath,
                seededHistory = seededHistory,
                maxHistoryMessages = seededHistory.size,
                stripReasoningFromHistory = false,
                prompt = SHORT_REPLY_PROMPT,
            )

        println("[ChatSessionLinuxE2ETest] smallWindowContext=${smallWindow.contextAfterTurn}")
        println("[ChatSessionLinuxE2ETest] largeWindowContext=${largeWindow.contextAfterTurn}")

        assertTrue("Small-window response should not be blank", smallWindow.response.isNotBlank())
        assertTrue("Large-window response should not be blank", largeWindow.response.isNotBlank())
        assertEquals(seededHistory.size + 1, smallWindow.history.size)
        assertEquals(seededHistory.size + 1, largeWindow.history.size)
        assertTrue(
            "A smaller maxHistoryMessages value should replay fewer tokens into the native context. small=${smallWindow.contextAfterTurn}, large=${largeWindow.contextAfterTurn}",
            smallWindow.contextAfterTurn < largeWindow.contextAfterTurn,
        )
    }

    @Test
    fun `desktop end-to-end chat session strips think tags from replayed assistant history`() = runBlocking {
        val modelPath = resolveModelPath()
        val seededHistory =
            listOf(
                ChatMessage(Role.USER, buildLongText("user-context", 12)),
                ChatMessage(
                    Role.ASSISTANT,
                    "<think>${buildLongText("analysis-token", 48)}</think> visible-answer",
                ),
                ChatMessage(Role.USER, buildLongText("follow-up", 8)),
            )

        val stripped =
            runSeededPartialTurn(
                modelPath = modelPath,
                seededHistory = seededHistory,
                maxHistoryMessages = seededHistory.size,
                stripReasoningFromHistory = true,
                prompt = SHORT_REPLY_PROMPT,
            )
        val unstripped =
            runSeededPartialTurn(
                modelPath = modelPath,
                seededHistory = seededHistory,
                maxHistoryMessages = seededHistory.size,
                stripReasoningFromHistory = false,
                prompt = SHORT_REPLY_PROMPT,
            )

        println("[ChatSessionLinuxE2ETest] strippedContext=${stripped.contextAfterTurn}")
        println("[ChatSessionLinuxE2ETest] unstrippedContext=${unstripped.contextAfterTurn}")

        assertTrue("Stripped-history response should not be blank", stripped.response.isNotBlank())
        assertTrue("Unstripped-history response should not be blank", unstripped.response.isNotBlank())
        assertTrue(
            "stripReasoningFromHistory should reduce native context usage when replaying assistant messages. stripped=${stripped.contextAfterTurn}, unstripped=${unstripped.contextAfterTurn}",
            stripped.contextAfterTurn < unstripped.contextAfterTurn,
        )
    }

    private suspend fun runFreshTurn(
        modelPath: String,
        systemPrompt: String?,
        prompt: String,
    ): SessionRun {
        val smol = loadSmol(modelPath)
        try {
            val session = ChatSession(smol, ChatSessionConfig(systemPrompt = systemPrompt))
            val response = session.sendMessage(prompt)
            return SessionRun(
                response = response,
                contextAfterTurn = smol.getContextLengthUsed(),
                history = session.historySnapshot(),
            )
        } finally {
            smol.close()
        }
    }

    private suspend fun runStreamedTurn(modelPath: String, prompt: String): SessionRun {
        val smol = loadSmol(modelPath)
        try {
            val session = ChatSession(smol, ChatSessionConfig(systemPrompt = LONG_SYSTEM_PROMPT))
            val response = session.sendMessageStream(prompt).toList().joinToString(separator = "")
            return SessionRun(
                response = response,
                contextAfterTurn = smol.getContextLengthUsed(),
                history = session.historySnapshot(),
            )
        } finally {
            smol.close()
        }
    }

    private suspend fun runSeededTurn(
        modelPath: String,
        seededHistory: List<ChatMessage>,
        maxHistoryMessages: Int,
        stripReasoningFromHistory: Boolean,
        prompt: String,
    ): SessionRun {
        val smol = loadSmol(modelPath)
        try {
            val session =
                ChatSession(
                    smol,
                    ChatSessionConfig(
                        maxHistoryMessages = maxHistoryMessages,
                        stripReasoningFromHistory = stripReasoningFromHistory,
                        systemPrompt = COMPACT_RESPONSE_SYSTEM_PROMPT,
                    ),
                )
            seedHistory(session, seededHistory)
            val response = session.sendMessage(prompt)
            return SessionRun(
                response = response,
                contextAfterTurn = smol.getContextLengthUsed(),
                history = session.historySnapshot(),
            )
        } finally {
            smol.close()
        }
    }

    private suspend fun runSeededPartialTurn(
        modelPath: String,
        seededHistory: List<ChatMessage>,
        maxHistoryMessages: Int,
        stripReasoningFromHistory: Boolean,
        prompt: String,
    ): SessionRun {
        val smol = loadSmol(modelPath)
        try {
            val session =
                ChatSession(
                    smol,
                    ChatSessionConfig(
                        maxHistoryMessages = maxHistoryMessages,
                        stripReasoningFromHistory = stripReasoningFromHistory,
                        systemPrompt = COMPACT_RESPONSE_SYSTEM_PROMPT,
                    ),
                )
            seedHistory(session, seededHistory)
            val response = session.sendMessageStream(prompt).take(1).toList().joinToString(separator = "")
            return SessionRun(
                response = response,
                contextAfterTurn = smol.getContextLengthUsed(),
                history = session.historySnapshot(),
            )
        } finally {
            smol.close()
        }
    }

    private suspend fun loadSmol(modelPath: String): SmolLM {
        val smol = SmolLM(useVulkan = false)
        smol.load(
            modelPath,
            SmolLM.InferenceParams(
                contextSize = 4096,
                temperature = 0.0f,
                storeChats = false,
            ),
        )
        return smol
    }

    private fun resolveModelPath(): String {
        val configuredPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        val modelPath = configuredPath ?: DEFAULT_MODEL_CANDIDATES.firstOrNull { File(it).exists() }

        println("[ChatSessionLinuxE2ETest] modelPath=$modelPath")
        Assume.assumeTrue("No text test model specified in $MODEL_PATH_ENV", !modelPath.isNullOrBlank())

        val projectDir = System.getProperty("user.dir")
        val defaultLibPath = "$projectDir/llmedge/build/native/linux-x86_64/libsmollm.so"
        val libPath = System.getenv(LIB_PATH_ENV) ?: defaultLibPath
        println("[ChatSessionLinuxE2ETest] libPath=$libPath libExists=${File(libPath).exists()}")
        Assume.assumeTrue("Native smollm library not found at $libPath", File(libPath).exists())

        val file = File(modelPath!!)
        Assume.assumeTrue("Test model not found at ${file.absolutePath}", file.exists())
        return file.absolutePath
    }

    private fun buildSeededHistory(messageCount: Int): List<ChatMessage> {
        val seededHistory = mutableListOf<ChatMessage>()
        repeat(messageCount) { index ->
            val role = if (index % 2 == 0) Role.USER else Role.ASSISTANT
            seededHistory +=
                ChatMessage(
                    role = role,
                    content = buildLongText("history-${index + 1}", 12),
                )
        }
        return seededHistory
    }

    private fun buildLongText(token: String, repetitions: Int): String =
        List(repetitions) { "$token-$it" }.joinToString(separator = " ")

    @Suppress("UNCHECKED_CAST")
    private fun seedHistory(session: ChatSession, seededHistory: List<ChatMessage>) {
        val field = ChatSession::class.java.getDeclaredField("history")
        field.isAccessible = true
        val history = field.get(session) as MutableList<ChatMessage>
        history.clear()
        history.addAll(seededHistory)
    }

    private data class SessionRun(
        val response: String,
        val contextAfterTurn: Int,
        val history: List<ChatMessage>,
    )

    private companion object {
        private const val MODEL_PATH_ENV = "LLMEDGE_TEST_TEXT_MODEL_PATH"
        private const val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
        private val DEFAULT_MODEL_CANDIDATES =
            listOf(
                "models/SmolLM2-135M-Instruct-Q8_0.gguf",
                "models/SmolLM2-1.7B-Instruct-Q8_0.gguf",
            )

        private const val NATURAL_PROMPT = "Give a short, friendly greeting."
        private const val SHORT_REPLY_PROMPT = "Reply with one short word."
        private val LONG_SYSTEM_PROMPT = buildString {
            append("You are a concise assistant. ")
            append(List(80) { "system-token-$it" }.joinToString(separator = " "))
        }
        private const val COMPACT_RESPONSE_SYSTEM_PROMPT =
            "Reply with one or two short words only."
    }
}
