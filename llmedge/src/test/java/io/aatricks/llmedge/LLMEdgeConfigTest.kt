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
            )

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
    }
}
