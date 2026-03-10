package io.aatricks.llmedge

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import io.aatricks.llmedge.text.runtime.SmolLM

class SmolLMThinkingModeTest {
    @Before
    fun setup() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        SmolLM.overrideNativeBridgeForTests { instance ->
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
                    kvCacheTypeK: Int,
                    kvCacheTypeV: Int,
                    nGpuLayers: Int,
                ): Long = 1L

                override fun setReasoningOptions(instance: SmolLM, modelPtr: Long, disableThinking: Boolean, reasoningBudget: Int) {}
                override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) {}
                override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 0f
                override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 0
                override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun nativeDecodePreparedEmbeddings(instance: SmolLM, modelPtr: Long, embdPath: String, metaPath: String, nBatch: Int): Boolean = true
                override fun close(instance: SmolLM, modelPtr: Long) {}
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {}
                override fun completionLoop(instance: SmolLM, modelPtr: Long): String = ""
                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String = ""
                override fun stopCompletion(instance: SmolLM, modelPtr: Long) {}
                override fun clearKvCache(instance: SmolLM, modelPtr: Long) {}
            }
        }
    }

    @After
    fun teardown() {
        SmolLM.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `default thinking mode is DEFAULT and toggles correctly`() {
        val smol = SmolLM.createLoadedForTests(1L)
        assertEquals(SmolLM.ThinkingMode.DEFAULT, smol.getThinkingMode())

        smol.setThinkingEnabled(false)
        assertEquals(SmolLM.ThinkingMode.DISABLED, smol.getThinkingMode())
        assertFalse(smol.isThinkingEnabled())

        smol.setThinkingEnabled(true)
        assertEquals(SmolLM.ThinkingMode.DEFAULT, smol.getThinkingMode())
        assertTrue(smol.isThinkingEnabled())
    }

    @Test
    fun `setReasoningBudget zero disables thinking`() {
        val smol = SmolLM.createLoadedForTests(1L)
        smol.setReasoningBudget(0)
        assertEquals(SmolLM.ThinkingMode.DISABLED, smol.getThinkingMode())
        assertEquals(0, smol.getReasoningBudget())
    }
}
