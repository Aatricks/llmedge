package io.aatricks.llmedge.rag

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InMemoryVectorStoreTest {
    @Test
    fun `upsert replaces matching id`() {
        val store = InMemoryVectorStore()

        store.upsert(VectorEntry("id", "old", floatArrayOf(1f, 0f)))
        store.upsert(VectorEntry("id", "new", floatArrayOf(0f, 1f)))

        assertEquals(1, store.size())
        assertEquals("new", store.head(1).single().text)
    }

    @Test
    fun `topKWithScores handles zero vectors without NaN`() {
        val store = InMemoryVectorStore()
        store.upsert(VectorEntry("id", "text", floatArrayOf(0f, 0f, 0f)))

        val score = store.topKWithScores(floatArrayOf(0f, 0f, 0f), 1).single().second

        assertEquals(0f, score)
    }

    @Test
    fun `save and load round trip entries`() {
        val persistFile = File.createTempFile("rag-store", ".json")
        persistFile.deleteOnExit()

        val writer = InMemoryVectorStore(persistFile)
        writer.addAll(
            listOf(
                VectorEntry("one", "hello", floatArrayOf(1f, 2f)),
                VectorEntry("two", "world", floatArrayOf(3f, 4f)),
            ),
        )
        writer.save()

        val reader = InMemoryVectorStore(persistFile)
        reader.load()

        assertEquals(2, reader.size())
        assertTrue(reader.head(2).map { it.id }.containsAll(listOf("one", "two")))
    }
}