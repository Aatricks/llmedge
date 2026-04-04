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
        val entries =
            chunks.map { chunk ->
                VectorEntry(
                    id = UUID.randomUUID().toString(),
                    text = chunk,
                    embedding = embeddingProvider.encode(chunk),
                )
            }
        vectorStore.addAll(entries)
        vectorStore.save()
        entries.size
    }
}
