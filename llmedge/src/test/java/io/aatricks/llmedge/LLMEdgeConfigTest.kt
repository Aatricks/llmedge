package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class LLMEdgeConfigTest {
    @Test
    fun `nested config is the single source of truth`() {
        val config =
            LLMEdgeConfig(
                execution = ExecutionConfig(inferenceThreads = 7),
                text =
                    TextRuntimeConfig(
                        cache = RuntimeCacheConfig(maxEntries = 3, maxMemoryMb = 768),
                        useVulkan = false,
                        promptThreads = 6,
                        generationThreads = 2,
                        batchSize = 9,
                        streamBatchSize = 5,
                        useFlashAttention = false,
                    ),
                image =
                    ImageRuntimeConfig(
                        cache = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 8192),
                        preferPerformanceMode = true,
                    ),
                vision =
                    VisionRuntimeConfig(
                        cache = RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 1536),
                        useVulkan = false,
                        promptThreads = 5,
                        generationThreads = 3,
                        useFlashAttention = false,
                    ),
            )

        assertEquals(7, config.execution.inferenceThreads)
        assertEquals(3, config.text.cache.maxEntries)
        assertEquals(768L, config.text.cache.maxMemoryMb)
        assertFalse(config.text.useVulkan)
        assertEquals(6, config.text.promptThreads)
        assertEquals(2, config.text.generationThreads)
        assertEquals(9, config.text.batchSize)
        assertEquals(5, config.text.streamBatchSize)
        assertFalse(config.text.useFlashAttention)
        assertEquals(8192L, config.image.cache.maxMemoryMb)
        assertTrue(config.image.preferPerformanceMode)
        assertEquals(1536L, config.vision.cache.maxMemoryMb)
        assertFalse(config.vision.useVulkan)
        assertEquals(5, config.vision.promptThreads)
        assertEquals(3, config.vision.generationThreads)
        assertFalse(config.vision.useFlashAttention)
    }

    @Test
    fun `runtime cache config validates positive limits`() {
        try {
            RuntimeCacheConfig(maxEntries = 0, maxMemoryMb = 256)
            fail("Expected RuntimeCacheConfig to reject zero cache entries")
        } catch (_: IllegalArgumentException) {
        }

        try {
            RuntimeCacheConfig(maxEntries = 1, maxMemoryMb = 0)
            fail("Expected RuntimeCacheConfig to reject zero memory budget")
        } catch (_: IllegalArgumentException) {
        }

        try {
            ExecutionConfig(inferenceThreads = 0)
            fail("Expected ExecutionConfig to reject zero inference threads")
        } catch (_: IllegalArgumentException) {
        }
    }
}
