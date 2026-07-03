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

    @Test
    fun `split handles CJK fallback correctly`() {
        val splitter = TextSplitter(chunkSize = 10, chunkOverlap = 2)
        val cjkText = "一一二二三三四四五五六六七七八八九九十十百百千千万"
        val chunks = splitter.split(cjkText)

        assertEquals(3, chunks.size)
        assertEquals("一一二二三三四四五五", chunks[0])
        assertEquals("五五六六七七八八九九", chunks[1])
        assertEquals("九九十十百百千千万", chunks[2])
    }
}