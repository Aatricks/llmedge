package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.text.runtime.SmolLM

class SmolLMDataClassesTest {

    @Test
    fun `SmolLM InferenceParams data class works correctly`() {
        val params = SmolLM.InferenceParams(
            minP = 0.05f,
            temperature = 0.7f,
            storeChats = false,
            contextSize = 2048L,
            chatTemplate = "test template",
            numThreads = 8,
            useMmap = false,
            useMlock = true,
            thinkingMode = SmolLM.ThinkingMode.DISABLED,
            reasoningBudget = 200
        )

        assertEquals(0.05f, params.minP)
        assertEquals(0.7f, params.temperature)
        assertEquals(false, params.storeChats)
        assertEquals(2048L, params.contextSize)
        assertEquals("test template", params.chatTemplate)
        assertEquals(8, params.numThreads)
        assertEquals(false, params.useMmap)
        assertEquals(true, params.useMlock)
        assertEquals(SmolLM.ThinkingMode.DISABLED, params.thinkingMode)
        assertEquals(200, params.reasoningBudget)
    }

    @Test
    fun `SmolLM InferenceParams default values work correctly`() {
        val params = SmolLM.InferenceParams()

        assertEquals(0.1f, params.minP)
        assertEquals(0.8f, params.temperature)
        assertEquals(true, params.storeChats)
        assertEquals(null, params.contextSize)
        assertEquals(null, params.chatTemplate)
        assertEquals(4, params.numThreads)
        assertEquals(true, params.useMmap)
        assertEquals(false, params.useMlock)
        assertEquals(SmolLM.ThinkingMode.DEFAULT, params.thinkingMode)
        assertEquals(null, params.reasoningBudget)
    }

    @Test
    fun `SmolLM GenerationMetrics data class works correctly`() {
        val metrics = SmolLM.GenerationMetrics(
            tokensPerSecond = 15.5f,
            tokenCount = 150L,
            elapsedMicros = 10000000L
        )

        assertEquals(15.5f, metrics.tokensPerSecond)
        assertEquals(150L, metrics.tokenCount)
        assertEquals(10000000L, metrics.elapsedMicros)
        assertEquals(10000.0, metrics.elapsedMillis, 0.001)
        assertEquals(10.0, metrics.elapsedSeconds, 0.001)
    }

    @Test
    fun `SmolLM DefaultInferenceParams works correctly`() {
        val defaultParams = SmolLM.DefaultInferenceParams

        assertNotNull(defaultParams)
        assertEquals(1024L, defaultParams.contextSize)
        assert(defaultParams.chatTemplate.contains("SmolLM"))
        assert(defaultParams.chatTemplate.contains("assistant"))
    }

    @Test
    fun `SmolLM ThinkingMode enum values work correctly`() {
        assertEquals(2, SmolLM.ThinkingMode.entries.size)
        assert(SmolLM.ThinkingMode.entries.contains(SmolLM.ThinkingMode.DEFAULT))
        assert(SmolLM.ThinkingMode.entries.contains(SmolLM.ThinkingMode.DISABLED))
    }

    @Test
    fun `SmolLM ThinkingMode DISABLED has correct properties`() {
        val mode = SmolLM.ThinkingMode.DISABLED

        assertEquals(true, mode.disableReasoning)
        assertEquals(0, mode.reasoningBudget)
    }

    @Test
    fun `SmolLM ThinkingMode DEFAULT has correct properties`() {
        val mode = SmolLM.ThinkingMode.DEFAULT

        assertEquals(false, mode.disableReasoning)
        assertEquals(-1, mode.reasoningBudget)
    }
}