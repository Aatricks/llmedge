package io.aatricks.llmedge.rag

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InMemoryVectorStoreTest {
    @Test
    fun `concurrent indexing and querying does not corrupt the store`() {
        val store = InMemoryVectorStore()
        val errors = java.util.concurrent.ConcurrentLinkedQueue<Throwable>()
        val iterations = 200

        val writer = Thread {
            try {
                repeat(iterations) { i ->
                    store.addAll(
                        (0 until 5).map { j ->
                            VectorEntry("w$i-$j", "text", FloatArray(16) { (i + j).toFloat() })
                        },
                    )
                }
            } catch (t: Throwable) {
                errors.add(t)
            }
        }
        val reader = Thread {
            try {
                repeat(iterations) {
                    store.topKWithScores(FloatArray(16) { 1f }, 3)
                    store.size()
                }
            } catch (t: Throwable) {
                errors.add(t)
            }
        }
        writer.start()
        reader.start()
        writer.join()
        reader.join()

        assertTrue("Concurrent access failed: ${errors.firstOrNull()}", errors.isEmpty())
        assertEquals(iterations * 5, store.size())
    }

    @Test
    fun `legacy json persist file migrates on load`() {
        val persistFile = File.createTempFile("rag-store", ".json")
        persistFile.deleteOnExit()
        persistFile.writeText(
            """[{"id":"one","text":"hello","embedding":[1.0,2.0]},""" +
                """{"id":"two","text":"world","embedding":[3.0,4.0]}]""",
        )

        val store = InMemoryVectorStore(persistFile)
        store.load()

        assertEquals(2, store.size())
        assertEquals("hello", store.head(1).single().text)

        // A save rewrites in the binary format and the store still round-trips.
        store.save()
        val reloaded = InMemoryVectorStore(persistFile)
        reloaded.load()
        assertEquals(2, reloaded.size())
    }

    @Test
    fun `corrupt persist file is discarded instead of breaking load`() {
        val persistFile = File.createTempFile("rag-store", ".json")
        persistFile.deleteOnExit()
        persistFile.writeText("{ this is not valid json")

        val store = InMemoryVectorStore(persistFile)
        store.load()

        assertTrue(store.isEmpty())
        // The corrupt file is removed so the next save starts clean.
        assertTrue(!persistFile.exists())
    }

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