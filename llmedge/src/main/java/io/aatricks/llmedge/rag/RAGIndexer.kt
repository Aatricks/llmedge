package io.aatricks.llmedge.rag

import android.content.Context
import android.net.Uri
import android.util.Log
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal class RAGIndexer(
    private val context: Context,
    private val splitter: TextSplitter,
    private val embeddingProvider: EmbeddingProvider,
    private val vectorStore: InMemoryVectorStore,
) {
    suspend fun indexPdf(uri: Uri): Int = withContext(Dispatchers.IO) {
        val text = PDFReader.readAllText(context, uri).trim()
        Log.d(RAGEngine.TAG, "PDF extracted chars=${text.length}")
        val chunks = splitter.split(text)
        Log.d(RAGEngine.TAG, "Chunk count=${chunks.size}")
        if (chunks.isEmpty()) {
            Log.w(RAGEngine.TAG, "No text extracted from PDF at $uri. The document may be image-only. Consider running OCR.")
        }
        val entries =
            chunks.mapIndexed { idx, chunk ->
                VectorEntry(
                    id = getDeterministicId(uri.toString(), idx, chunk),
                    text = chunk,
                    embedding = embeddingProvider.encode(chunk),
                )
            }
        vectorStore.addAll(entries)
        vectorStore.save()
        entries.size
    }

    private fun getDeterministicId(documentUri: String, chunkIndex: Int, chunkText: String): String {
        val input = "$documentUri:$chunkIndex:$chunkText"
        val md = java.security.MessageDigest.getInstance("SHA-256")
        val bytes = md.digest(input.toByteArray(Charsets.UTF_8))
        return bytes.joinToString("") { "%02x".format(it) }
    }
}
