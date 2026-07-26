package io.aatricks.llmedge.rag

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.io.File
import kotlinx.coroutines.test.runTest

class RAGTest {

    @Test
    fun `TextSplitter splits text correctly`() {
        val splitter = TextSplitter(chunkSize = 10, chunkOverlap = 3)

        val text = "This is a test document with multiple sentences."
        val chunks = splitter.split(text)

        assertNotNull(chunks)
        assertEquals(1, chunks.size) // Should split into chunks
    }

    @Test
    fun `TextSplitter handles empty text`() {
        val splitter = TextSplitter(chunkSize = 10, chunkOverlap = 3)

        val chunks = splitter.split("")

        assertNotNull(chunks)
        assertEquals(0, chunks.size)
    }

    @Test
    fun `TextSplitter handles short text`() {
        val splitter = TextSplitter(chunkSize = 10, chunkOverlap = 3)

        val text = "Short"
        val chunks = splitter.split(text)

        assertNotNull(chunks)
        assertEquals(1, chunks.size)
        assertEquals("Short", chunks[0])
    }

    @Test
    fun `InMemoryVectorStore constructor initializes correctly`() {
        val store = InMemoryVectorStore(File("/tmp/test_store"))

        assertNotNull(store)
    }

    @Test
    fun `VectorEntry constructor initializes correctly`() {
        val embedding = floatArrayOf(0.1f, 0.2f, 0.3f)
        val entry = VectorEntry("test-id", "test text", embedding)

        assertEquals("test-id", entry.id)
        assertEquals("test text", entry.text)
        assertEquals(embedding, entry.embedding)
    }

    @Test
    fun `EmbeddingConfig constructor initializes correctly`() {
        val config = EmbeddingConfig(
            modelAssetPath = "/models/embedding",
            tokenizerAssetPath = "/models/tokenizer",
            useTokenTypeIds = false,
            outputTensorName = "text_embeddings",
            useFP16 = false,
            useXNNPack = false
        )

        assertEquals("/models/embedding", config.modelAssetPath)
        assertEquals("/models/tokenizer", config.tokenizerAssetPath)
        assertEquals(false, config.useTokenTypeIds)
        assertEquals("text_embeddings", config.outputTensorName)
        assertEquals(false, config.useFP16)
        assertEquals(false, config.useXNNPack)
    }

    @Test
    fun `local embedding files are used without copying packaged assets`() = runTest {
        val directory = createTempDir(prefix = "embedding-files-")
        val modelFile = File(directory, "model.onnx").apply { writeText("model") }
        val tokenizerFile = File(directory, "tokenizer.json").apply { writeText("{}") }
        val copiedModel = File(directory, "copied-model.onnx")
        val config = EmbeddingConfig.fromFiles(modelFile, tokenizerFile)

        val resolvedModel =
            EmbeddingFileSource.parse(config.modelAssetPath).materialize(copiedModel) {
                error("AssetManager must not be used for local embedding files")
            }

        assertEquals(modelFile.canonicalFile, resolvedModel.canonicalFile)
        assertFalse(copiedModel.exists())
    }

    @Test
    fun `InMemoryVectorStore operations work correctly`() {
        val store = InMemoryVectorStore(File("/tmp/test_store"))

        // Test empty store
        assertEquals(true, store.isEmpty())
        assertEquals(0, store.size())

        // Add entries
        val entry1 = VectorEntry("id1", "text1", floatArrayOf(1f, 2f, 3f))
        val entry2 = VectorEntry("id2", "text2", floatArrayOf(4f, 5f, 6f))
        store.upsert(entry1)
        store.upsert(entry2)

        assertEquals(false, store.isEmpty())
        assertEquals(2, store.size())

        // Test head
        val head = store.head(1)
        assertEquals(1, head.size)
        assertEquals("id1", head[0].id)
    }

    @Test
    fun `InMemoryVectorStore topK works correctly`() {
        val store = InMemoryVectorStore(File("/tmp/test_store"))

        // Add test entries
        store.upsert(VectorEntry("id1", "text1", floatArrayOf(1f, 0f, 0f)))
        store.upsert(VectorEntry("id2", "text2", floatArrayOf(0f, 1f, 0f)))
        store.upsert(VectorEntry("id3", "text3", floatArrayOf(0f, 0f, 1f)))

        // Query with vector similar to id1
        val query = floatArrayOf(0.9f, 0.1f, 0.1f)
        val results = store.topK(query, 2)

        assertEquals(2, results.size)
        assertEquals("id1", results[0].id) // Should be most similar
    }

    @Test
    fun `InMemoryVectorStore topKWithScores works correctly`() {
        val store = InMemoryVectorStore(File("/tmp/test_store"))

        store.upsert(VectorEntry("id1", "text1", floatArrayOf(1f, 0f, 0f)))

        val query = floatArrayOf(1f, 0f, 0f)
        val results = store.topKWithScores(query, 1)

        assertEquals(1, results.size)
        val (entry, score) = results[0]
        assertEquals("id1", entry.id)
        assertEquals(1.0f, score) // Perfect match
    }

    @Test
    fun `InMemoryVectorStore addAll works correctly`() {
        val store = InMemoryVectorStore(File("/tmp/test_store"))

        val entries = listOf(
            VectorEntry("id1", "text1", floatArrayOf(1f, 2f, 3f)),
            VectorEntry("id2", "text2", floatArrayOf(4f, 5f, 6f))
        )

        store.addAll(entries)

        assertEquals(2, store.size())
        assertEquals(false, store.isEmpty())
    }
}
