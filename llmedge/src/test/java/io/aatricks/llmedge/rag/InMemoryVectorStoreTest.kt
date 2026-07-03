package io.aatricks.llmedge.rag

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import io.mockk.mockk

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

    @Test
    fun `oversized-blob save rejection`() {
        val persistFile = File.createTempFile("rag-store-oversized", ".json")
        persistFile.deleteOnExit()
        val store = InMemoryVectorStore(persistFile)
        val largeText = "a".repeat(16 * 1024 * 1024 + 1)
        val entry = VectorEntry("id", largeText, floatArrayOf(1f, 2f))
        
        var threwInUpsert = false
        try {
            store.upsert(entry)
        } catch (e: IllegalArgumentException) {
            threwInUpsert = true
        }
        assertTrue("Upsert should reject oversized text", threwInUpsert)
    }

    @Test
    fun `oversized-blob addAll rejection`() {
        val store = InMemoryVectorStore()
        val largeText = "a".repeat(16 * 1024 * 1024 + 1)
        val entry = VectorEntry("id", largeText, floatArrayOf(1f, 2f))
        var threwInAddAll = false
        try {
            store.addAll(listOf(entry))
        } catch (e: IllegalArgumentException) {
            threwInAddAll = true
        }
        assertTrue("addAll should reject oversized text", threwInAddAll)
    }

    @Test
    fun `dim-mismatch upsert rejection and cosine-mismatch returns 0`() {
        val store = InMemoryVectorStore()
        store.upsert(VectorEntry("id1", "text1", floatArrayOf(1f, 2f)))
        
        var threwInUpsert = false
        try {
            store.upsert(VectorEntry("id2", "text2", floatArrayOf(1f, 2f, 3f)))
        } catch (e: IllegalArgumentException) {
            threwInUpsert = true
        }
        assertTrue("Should reject entry with different dimension", threwInUpsert)
        
        val scores = store.topKWithScores(floatArrayOf(1f, 2f, 3f), 1)
        if (scores.isNotEmpty()) {
            assertEquals(0f, scores.first().second)
        }
    }

    @Test
    fun `deterministic chunk ids on re-index`() = kotlinx.coroutines.runBlocking {
        io.mockk.mockkStatic(android.util.Log::class)
        io.mockk.every { android.util.Log.d(any<String>(), any<String>()) } returns 0
        io.mockk.every { android.util.Log.w(any<String>(), any<String>()) } returns 0

        val context = mockk<android.content.Context>(relaxed = true)
        val uri = mockk<android.net.Uri>()
        io.mockk.every { uri.toString() } returns "content://mock/test.pdf"

        io.mockk.mockkObject(PDFReader)
        io.mockk.coEvery { PDFReader.readAllText(context, uri) } returns "Some test text for PDF"

        val embeddingProvider = mockk<EmbeddingProvider>()
        io.mockk.coEvery { embeddingProvider.encode(any()) } returns floatArrayOf(0.1f, 0.2f)

        val persistFile = File.createTempFile("rag-store-deterministic", ".json")
        persistFile.deleteOnExit()
        val store = InMemoryVectorStore(persistFile)

        val indexer = RAGIndexer(context, TextSplitter(), embeddingProvider, store)

        val firstCount = indexer.indexPdf(uri)
        assertEquals(1, firstCount)
        assertEquals(1, store.size())
        val firstEntry = store.head(1).first()

        val secondCount = indexer.indexPdf(uri)
        assertEquals(1, secondCount)
        assertEquals(1, store.size())
        val secondEntry = store.head(1).first()

        assertEquals(firstEntry.id, secondEntry.id)
        
        io.mockk.unmockkObject(PDFReader)
        io.mockk.unmockkStatic(android.util.Log::class)
    }

    @Test
    fun `shared store instance across two RAGEngine constructions on the same path`() {
        val file = File("/tmp/test_store_shared")
        val store1 = RAGEngine.getOrCreateStore(file)
        val store2 = RAGEngine.getOrCreateStore(file)
        assertTrue("Should share the same vector store instance", store1 === store2)
    }
}