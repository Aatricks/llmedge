package io.aatricks.llmedge

import kotlinx.coroutines.flow.take
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
        // 1 call returns the text, then the stepper polls a bounded number of
        // consecutive empty batches (legacy bridges return "" both while buffering
        // UTF-8 and at end-of-stream) before treating the stream as finished.
        assertEquals(5, bridge.completionLoopBatchCalls)

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

    /** Bridge that streams tokens forever until stopCompletion is called. */
    private class EndlessBridge : SmolLM.NativeBridge {
        var stopCalls = 0

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
        ): Long = 1L

        override fun setReasoningOptions(instance: SmolLM, modelPtr: Long, disableThinking: Boolean, reasoningBudget: Int) {}
        override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) {}
        override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 1f
        override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 1
        override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 1L
        override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 1
        override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L
        override fun nativeDecodePreparedEmbeddings(instance: SmolLM, modelPtr: Long, embdPath: String, metaPath: String, nBatch: Int): Boolean = true
        override fun close(instance: SmolLM, modelPtr: Long) {}
        override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {}
        override fun completionLoop(instance: SmolLM, modelPtr: Long): String = "tok "
        override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String = "tok "
        override fun stopCompletion(instance: SmolLM, modelPtr: Long) {
            stopCalls++
        }
        override fun clearKvCache(instance: SmolLM, modelPtr: Long) {}
    }

    @Test
    fun test_flow_cancellation_via_take_is_not_wrapped_as_failure() = runBlocking {
        val bridge = EndlessBridge()
        SmolLM.overrideNativeBridgeForTests { _ -> bridge }
        val smol = SmolLM.createLoadedForTests(1L, useVulkan = false)

        // take() cancels the flow after 3 emissions; this must terminate the
        // stream cleanly, not surface as InferenceFailedException.
        val pieces = smol.getResponseAsFlow("q", kotlinx.coroutines.Dispatchers.Unconfined, 1)
            .take(3)
            .toList()

        assertEquals(3, pieces.size)
        assertEquals(1, bridge.stopCalls)
    }

    @Test
    fun test_native_failure_in_completion_loop_is_wrapped() = runBlocking {
        val bridge = object : SmolLM.NativeBridge by EndlessBridge() {
            override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String =
                throw IllegalStateException("native loop failed")
            override fun completionLoop(instance: SmolLM, modelPtr: Long): String =
                throw IllegalStateException("native loop failed")
        }
        SmolLM.overrideNativeBridgeForTests { _ -> bridge }
        val smol = SmolLM.createLoadedForTests(1L, useVulkan = false)

        var thrown: Throwable? = null
        try {
            smol.getResponseAsFlow("q", kotlinx.coroutines.Dispatchers.Unconfined, 1).toList()
        } catch (e: Throwable) {
            thrown = e
        }
        assertEquals(
            io.aatricks.llmedge.core.InferenceFailedException::class.java,
            thrown?.javaClass,
        )
    }
}
