package io.aatricks.llmedge

import java.io.ByteArrayOutputStream
import java.io.PrintStream
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class ChatSessionTest {
    private class RecordingBridge : SmolLM.NativeBridge {
        val injectedTurns = mutableListOf<List<Pair<String, String>>>()
        val completionPrompts = mutableListOf<String>()
        var clearKvCacheCount = 0

        private val pendingResponses = ArrayDeque<ArrayDeque<String>>()
        private val pendingMessages = mutableListOf<Pair<String, String>>()
        private var activeResponse = ArrayDeque<String>()

        fun enqueueResponse(vararg pieces: String) {
            pendingResponses.addLast(ArrayDeque(pieces.toList() + "[EOG]"))
        }

        override fun loadModel(
            instance: SmolLM,
            modelPath: String,
            minP: Float,
            temperature: Float,
            storeChats: Boolean,
            contextSize: Long,
            chatTemplate: String,
            nThreads: Int,
            useMmap: Boolean,
            useMlock: Boolean,
            useVulkan: Boolean,
            useFlashAttn: Boolean,
        ): Long = 1L

        override fun setReasoningOptions(
            instance: SmolLM,
            modelPtr: Long,
            disableThinking: Boolean,
            reasoningBudget: Int,
        ) {}

        override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) {
            pendingMessages.add(role to message)
        }

        override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 0f

        override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 0L

        override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 0L

        override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 0

        override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L

        override fun nativeDecodePreparedEmbeddings(
            instance: SmolLM,
            modelPtr: Long,
            embdPath: String,
            metaPath: String,
            nBatch: Int,
        ): Boolean = true

        override fun close(instance: SmolLM, modelPtr: Long) {}

        override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
            completionPrompts.add(prompt)
            injectedTurns.add(pendingMessages.toList())
            pendingMessages.clear()
            activeResponse = pendingResponses.removeFirstOrNull() ?: ArrayDeque(listOf("[EOG]"))
        }

        override fun completionLoop(instance: SmolLM, modelPtr: Long): String = activeResponse.removeFirst()

        override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String =
            completionLoop(instance, modelPtr)

        override fun stopCompletion(instance: SmolLM, modelPtr: Long) {}

        override fun clearKvCache(instance: SmolLM, modelPtr: Long) {
            clearKvCacheCount++
        }
    }

    @Before
    fun setup() {
        System.setProperty("llmedge.disableNativeLoad", "true")
    }

    @After
    fun teardown() {
        SmolLM.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `stripThinkBlocks removes multiline and dangling think tags`() {
        val raw = "Before\n<think>hidden\nreasoning</think>\nAfter<think>unfinished"

        assertEquals("Before\n\nAfter", raw.stripThinkBlocks())
    }

    @Test
    fun `sendMessage clears context and strips reasoning when replaying history`() = runTest {
        val bridge = RecordingBridge()
        val session =
            createSession(
                bridge = bridge,
                config =
                    ChatSessionConfig(
                        maxHistoryMessages = 3,
                        systemPrompt = "System prompt",
                    ),
            )

        bridge.enqueueResponse("<think>private</think> Visible answer")
        assertEquals("<think>private</think> Visible answer", session.sendMessage("Hello"))

        bridge.enqueueResponse("Second answer")
        assertEquals("Second answer", session.sendMessage("Follow up"))

        assertEquals(2, bridge.clearKvCacheCount)
        assertEquals(listOf("", ""), bridge.completionPrompts)
        assertEquals(
            listOf(
                listOf(
                    "system" to "System prompt",
                    "user" to "Hello",
                ),
                listOf(
                    "system" to "System prompt",
                    "user" to "Hello",
                    "assistant" to "Visible answer",
                    "user" to "Follow up",
                ),
            ),
            bridge.injectedTurns,
        )
    }

    @Test
    fun `sendMessage applies sliding window to individual messages`() = runTest {
        val bridge = RecordingBridge()
        val session = createSession(bridge = bridge, config = ChatSessionConfig(maxHistoryMessages = 2))

        bridge.enqueueResponse("A1")
        session.sendMessage("Q1")

        bridge.enqueueResponse("A2")
        session.sendMessage("Q2")

        bridge.enqueueResponse("A3")
        session.sendMessage("Q3")

        assertEquals(
            listOf(
                "assistant" to "A2",
                "user" to "Q3",
            ),
            bridge.injectedTurns[2],
        )
    }

    @Test
    fun `sendMessageStream stores completed raw reply and replays stripped history`() = runTest {
        val bridge = RecordingBridge()
        val session =
            createSession(
                bridge = bridge,
                config = ChatSessionConfig(maxHistoryMessages = 4, systemPrompt = "System prompt"),
            )

        bridge.enqueueResponse("<think>private</think>", " Final answer")
        val chunks = session.sendMessageStream("Stream please").toList()

        assertEquals(listOf("<think>private</think>", " Final answer"), chunks)

        bridge.enqueueResponse("Done")
        session.sendMessage("Next question")

        assertEquals(
            listOf(
                "system" to "System prompt",
                "user" to "Stream please",
                "assistant" to "Final answer",
                "user" to "Next question",
            ),
            bridge.injectedTurns[1],
        )
        assertEquals(
            listOf(
                ChatMessage(Role.USER, "Stream please"),
                ChatMessage(Role.ASSISTANT, "<think>private</think> Final answer"),
                ChatMessage(Role.USER, "Next question"),
                ChatMessage(Role.ASSISTANT, "Done"),
            ),
            session.historySnapshot(),
        )
    }

    @Test
    fun `cancelled stream keeps user turn but does not append partial assistant reply`() = runTest {
        val bridge = RecordingBridge()
        val session = createSession(bridge = bridge)

        bridge.enqueueResponse("Hello", " world")
        assertEquals(listOf("Hello"), session.sendMessageStream("Interrupted").take(1).toList())

        bridge.enqueueResponse("Recovered")
        session.sendMessage("Retry")

        assertEquals(
            listOf(
                "user" to "Interrupted",
                "user" to "Retry",
            ),
            bridge.injectedTurns[1],
        )
        assertEquals(
            listOf(
                ChatMessage(Role.USER, "Interrupted"),
                ChatMessage(Role.USER, "Retry"),
                ChatMessage(Role.ASSISTANT, "Recovered"),
            ),
            session.historySnapshot(),
        )
    }

    @Test
    fun `sendMessage warns once when SmolLM was loaded with storeChats enabled`() = runTest {
        val bridge = RecordingBridge()
        val session = createSession(bridge = bridge, storeChats = true)
        val originalErr = System.err
        val errBuffer = ByteArrayOutputStream()
        System.setErr(PrintStream(errBuffer))

        try {
            bridge.enqueueResponse("One")
            session.sendMessage("Hello")

            bridge.enqueueResponse("Two")
            session.sendMessage("Again")
        } finally {
            System.setErr(originalErr)
        }

        val output = errBuffer.toString(Charsets.UTF_8.name())
        assertTrue(output.contains("storeChats = false"))
        assertTrue(session.hasStoreChatsConflictWarning)
        assertFalse(output.substringAfter("storeChats = false", "").contains("storeChats = false"))
    }

    private fun createSession(
        bridge: RecordingBridge,
        config: ChatSessionConfig = ChatSessionConfig(),
        storeChats: Boolean = false,
    ): ChatSession {
        SmolLM.overrideNativeBridgeForTests { _ -> bridge }
        val smol =
            SmolLM.createLoadedForTests(
                nativePtr = 1L,
                useVulkan = false,
                loadedParams = SmolLM.InferenceParams(storeChats = storeChats),
            )
        return ChatSession(smol, config)
    }
}
