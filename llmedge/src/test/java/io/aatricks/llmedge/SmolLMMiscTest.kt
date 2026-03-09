package io.aatricks.llmedge

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class SmolLMMiscTest {
    private class TestBridge : SmolLM.NativeBridge {
        var closeCalled = false

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

        override fun setReasoningOptions(instance: SmolLM, modelPtr: Long, disableThinking: Boolean, reasoningBudget: Int) { /* no-op */ }
        override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) { /* no-op */ }
        override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 12.5f
        override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 7
        override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 2_000_000L
        override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 314
        override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0xCAFEL
        override fun nativeDecodePreparedEmbeddings(instance: SmolLM, modelPtr: Long, embdPath: String, metaPath: String, nBatch: Int): Boolean = true
        override fun close(instance: SmolLM, modelPtr: Long) { closeCalled = true }
        override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) { /* no-op */ }
        override fun completionLoop(instance: SmolLM, modelPtr: Long): String = "[EOG]"
        override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String = "[EOG]"
        override fun stopCompletion(instance: SmolLM, modelPtr: Long) { /* no-op */ }
        override fun clearKvCache(instance: SmolLM, modelPtr: Long) { /* no-op */ }
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
    fun `isVulkanEnabled reflects constructor flag`() {
        val smolCpu = SmolLM(useVulkan = false)
        assertFalse(smolCpu.isVulkanEnabled())

        val smolGpu = SmolLM(useVulkan = true)
        assertTrue(smolGpu.isVulkanEnabled())
    }

    @Test
    fun `native pointer helpers and close work with stub bridge`() {
        val bridge = TestBridge()
        SmolLM.overrideNativeBridgeForTests { _ -> bridge }

        val smol = SmolLM.createLoadedForTests(123L)

        // Cover simple accessors that require a valid native handle
        assertEquals(12.5f, smol.getResponseGenerationSpeed(), 0.0001f)
        assertEquals(314, smol.getContextLengthUsed())
        assertEquals(0xCAFEL, smol.getNativeModelPointer())
        assertTrue(smol.decodePreparedEmbeddings("/tmp/embd.bin", "/tmp/meta.json", nBatch = 2))

        // Flip thinking mode to non-default, then ensure close() resets it
        smol.setThinkingEnabled(false)
        assertEquals(SmolLM.ThinkingMode.DISABLED, smol.getThinkingMode())

        smol.close()
        assertTrue(bridge.closeCalled)
        assertEquals(SmolLM.ThinkingMode.DEFAULT, smol.getThinkingMode())
        assertEquals(-1, smol.getReasoningBudget())
    }

    @Test
    fun `getLastGenerationMetrics prefers combined bridge path`() {
        SmolLM.overrideNativeBridgeForTests { _ ->
            object : SmolLM.NativeBridge {
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

                override fun setReasoningOptions(instance: SmolLM, modelPtr: Long, disableThinking: Boolean, reasoningBudget: Int) = Unit
                override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) = Unit
                override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = error("legacy speed path should not be used")
                override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = error("legacy token path should not be used")
                override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = error("legacy duration path should not be used")
                override fun getLastGenerationMetrics(instance: SmolLM, modelPtr: Long): SmolLM.GenerationMetrics =
                    SmolLM.GenerationMetrics(tokensPerSecond = 9.5f, tokenCount = 11L, elapsedMicros = 1234L)
                override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 0
                override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun nativeDecodePreparedEmbeddings(instance: SmolLM, modelPtr: Long, embdPath: String, metaPath: String, nBatch: Int): Boolean = true
                override fun close(instance: SmolLM, modelPtr: Long) = Unit
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) = Unit
                override fun completionLoop(instance: SmolLM, modelPtr: Long): String = "[EOG]"
                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String = "[EOG]"
                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit
                override fun clearKvCache(instance: SmolLM, modelPtr: Long) = Unit
            }
        }

        val smol = SmolLM.createLoadedForTests(99L)
        val metrics = smol.getLastGenerationMetrics()

        assertEquals(9.5f, metrics.tokensPerSecond, 0.0001f)
        assertEquals(11L, metrics.tokenCount)
        assertEquals(1234L, metrics.elapsedMicros)
    }

    @Test
    fun `estimated native memory helpers delegate to bridge`() {
        SmolLM.overrideNativeBridgeForTests { _ ->
            object : SmolLM.NativeBridge {
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

                override fun getEstimatedNativeMemoryBytes(instance: SmolLM, modelPtr: Long): Long = 321L
                override fun getEstimatedStateMemoryBytes(instance: SmolLM, modelPtr: Long): Long = 123L
                override fun setReasoningOptions(instance: SmolLM, modelPtr: Long, disableThinking: Boolean, reasoningBudget: Int) = Unit
                override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) = Unit
                override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 0f
                override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 0
                override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun nativeDecodePreparedEmbeddings(instance: SmolLM, modelPtr: Long, embdPath: String, metaPath: String, nBatch: Int): Boolean = true
                override fun close(instance: SmolLM, modelPtr: Long) = Unit
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) = Unit
                override fun completionLoop(instance: SmolLM, modelPtr: Long): String = "[EOG]"
                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String = "[EOG]"
                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit
                override fun clearKvCache(instance: SmolLM, modelPtr: Long) = Unit
            }
        }

        val smol = SmolLM.createLoadedForTests(77L)

        assertEquals(321L, smol.getEstimatedNativeMemoryBytes())
        assertEquals(123L, smol.getEstimatedStateMemoryBytes())
    }
}
