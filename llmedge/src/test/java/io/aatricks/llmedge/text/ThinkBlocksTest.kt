package io.aatricks.llmedge.text

import org.junit.Assert.assertEquals
import org.junit.Test

class ThinkBlocksTest {
    @Test
    fun `stripThinkBlocks removes multiline and dangling think tags`() {
        val raw = "Before\n<think>hidden\nreasoning</think>\nAfter<think>unfinished"

        assertEquals("Before\n\nAfter", raw.stripThinkBlocks())
    }
}
