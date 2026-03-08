package io.aatricks.llmedge.rag

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RAGPromptSupportTest {
    @Test
    fun `buildContextFromHits filters weak matches but falls back to top hit`() {
        val fallback =
            RAGPromptSupport.buildContextFromHits(
                listOf(VectorEntry("a", "useful fallback text", floatArrayOf()).let { it to 0.05f }),
            )

        assertTrue(fallback.contains("useful fallback text"))
        assertTrue(fallback.contains("score=0.050"))
    }

    @Test
    fun `buildContextFromHits keeps strong matches and separators`() {
        val context =
            RAGPromptSupport.buildContextFromHits(
                listOf(
                    VectorEntry("a", "first chunk", floatArrayOf()) to 0.91f,
                    VectorEntry("b", "second chunk", floatArrayOf()) to 0.42f,
                ),
            )

        assertTrue(context.contains("first chunk"))
        assertTrue(context.contains("second chunk"))
        assertTrue(context.contains("---"))
    }

    @Test
    fun `formatRetrievalPreview handles empty hits`() {
        assertEquals("(no hits)", RAGPromptSupport.formatRetrievalPreview(emptyList()))
    }
}