package io.aatricks.llmedge

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import io.aatricks.llmedge.text.runtime.SmolLM

class SmolLMInferenceTest {
    @Before
    fun setup() {
        // Ensure no native load on tests
        System.setProperty("llmedge.disableNativeLoad", "true")
    }

    @After
    fun teardown() {
        SmolLM.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun test_getResponse_and_flow_with_stubbed_native() = runBlocking {
        class TestBridge : SmolLM.NativeBridge {
            var messages = mutableListOf<Pair<String, String>>()
            private var queue = ArrayDeque<String>()
            var completionLoopCalls = 0
            var completionLoopBatchCalls = 0
            

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
                kvCacheTypeK: Int,
                kvCacheTypeV: Int,
                nGpuLayers: Int,
            ): Long {
                // return fake handle
                return 1L
            }

            override fun setReasoningOptions(instance: SmolLM, modelPtr: Long, disableThinking: Boolean, reasoningBudget: Int) {
                // no-op
            }

            override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) {
                messages.add(Pair(role, message))
            }

            override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 100f
            override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 4
            override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 1_000_000L
            override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 128
            override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L
            override fun nativeDecodePreparedEmbeddings(instance: SmolLM, modelPtr: Long, embdPath: String, metaPath: String, nBatch: Int): Boolean = true
            override fun close(instance: SmolLM, modelPtr: Long) {}
            override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                queue.addLast("Hello")
                queue.addLast(" ")
                queue.addLast("world")
                queue.addLast("[EOG]")
            }

            override fun completionLoop(instance: SmolLM, modelPtr: Long): String {
                completionLoopCalls++
                return queue.removeFirst()
            }
            override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String {
                completionLoopBatchCalls++
                val result = StringBuilder()
                repeat(maxTokens) {
                    if (queue.isEmpty()) return result.toString()
                    val piece = queue.removeFirst()
                    if (piece == "[EOG]") { if (result.isEmpty()) return "[EOG]" else return result.toString() }
                    result.append(piece)
                }
                return result.toString()
            }

            override fun stopCompletion(instance: SmolLM, modelPtr: Long) {}
            override fun clearKvCache(instance: SmolLM, modelPtr: Long) {}
        }
        val bridge = TestBridge()

        SmolLM.overrideNativeBridgeForTests { _ -> bridge }

        val smol = SmolLM.createLoadedForTests(1L, useVulkan = false)

        // Test getResponse
        val out = smol.getResponse("test prompt")
        assertEquals("Hello world", out)
        assertEquals(0, bridge.completionLoopCalls)
        assertEquals(2, bridge.completionLoopBatchCalls)

        // Test flow: the default stream batch groups a few tokens per native call,
        // so assert on the reassembled text rather than per-token emission.
        SmolLM.overrideNativeBridgeForTests { _ -> bridge }
        val flowList = smol.getResponseAsFlow("test prompt").toList()
        assertEquals("Hello world", flowList.joinToString(""))

        // With an explicit batchSize of 1 the flow still emits token by token.
        SmolLM.overrideNativeBridgeForTests { _ -> bridge }
        val perTokenList =
            smol.getResponseAsFlow("test prompt", kotlinx.coroutines.Dispatchers.Unconfined, 1).toList()
        assertEquals(listOf("Hello", " ", "world"), perTokenList)

        // Test metrics
        val metrics = smol.getLastGenerationMetrics()
        assertEquals(100f, metrics.tokensPerSecond, 0.001f)
        assertEquals(4L, metrics.tokenCount)
        assertEquals(1_000_000L, metrics.elapsedMicros)

        // Test chat messages
        smol.addUserMessage("Hi")
        smol.addSystemPrompt("System test")
        smol.addAssistantMessage("Assistant test")
        assertEquals(3, bridge.messages.size)
        assertEquals("user", bridge.messages[0].first)
        assertEquals("Hi", bridge.messages[0].second)
        assertEquals("system", bridge.messages[1].first)
        assertEquals("Assistant test", bridge.messages[2].second)
    }
}
