package io.aatricks.llmedge.rag

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal class RAGRetriever(
    private val embeddingProvider: EmbeddingProvider,
    private val vectorStore: InMemoryVectorStore,
) {
    suspend fun retrieve(question: String, topK: Int = 5): List<Pair<VectorEntry, Float>> =
        withContext(Dispatchers.IO) {
            val qEmb = embeddingProvider.encode(question)
            vectorStore.topKWithScores(qEmb, topK)
        }

    suspend fun contextFor(question: String, topK: Int = 5): String = withContext(Dispatchers.IO) {
        val hitsWithScores = retrieve(question, topK)
        if (hitsWithScores.isEmpty()) {
            return@withContext ""
        }
        RAGPromptSupport.buildContextFromHits(hitsWithScores)
    }

    suspend fun retrievalPreview(question: String, topK: Int = 5): String = withContext(Dispatchers.IO) {
        val hits = retrieve(question, topK)
        RAGPromptSupport.formatRetrievalPreview(hits)
    }
}
