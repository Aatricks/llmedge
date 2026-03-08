package io.aatricks.llmedge.rag

import org.junit.Assert.assertEquals
import org.junit.Test

class TextSplitterTest {
    @Test
    fun `split preserves overlap between adjacent chunks`() {
        val splitter = TextSplitter(chunkSize = 4, chunkOverlap = 1)

        val chunks = splitter.split("a b c d e f g")

        assertEquals(listOf("a b c d", "d e f g"), chunks)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `chunk overlap must be smaller than chunk size`() {
        TextSplitter(chunkSize = 4, chunkOverlap = 4)
    }
}