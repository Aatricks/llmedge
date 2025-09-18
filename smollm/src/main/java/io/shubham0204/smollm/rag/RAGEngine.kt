package io.shubham0204.smollm.rag

import android.content.Context
import android.net.Uri
import io.shubham0204.smollm.SmolLM
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

/**
 * Minimal on-device RAG pipeline wiring:
 * - Document load (PDF only for now)
 * - Chunking via TextSplitter
 * - Embeddings via Sentence-Embeddings
 * - Vector search (cosine)
 * - Prompt building and answer via SmolLM
 */
class RAGEngine(
    private val context: Context,
    private val smolLM: SmolLM,
    private val splitter: TextSplitter = TextSplitter(),
    embeddingConfig: EmbeddingConfig = EmbeddingConfig(),
) {
    private val embeddingProvider = EmbeddingProvider(context, embeddingConfig)
    @Volatile private var lastContext: String = ""
    private val vectorStore = InMemoryVectorStore(
        File(context.filesDir, "rag_store/index.json")
    )

    suspend fun init() {
        vectorStore.load()
        embeddingProvider.init()
    }

    suspend fun indexPdf(uri: Uri): Int = withContext(Dispatchers.Default) {
        val text = PDFReader.readAllText(context, uri).trim()
        Log.d(TAG, "PDF extracted chars=${text.length}")
        val chunks = splitter.split(text)
        Log.d(TAG, "Chunk count=${chunks.size}")
        val entries = chunks.mapIndexed { idx, chunk ->
            val id = UUID.randomUUID().toString()
            val emb = embeddingProvider.encode(chunk)
            VectorEntry(id = id, text = chunk, embedding = emb)
        }
        vectorStore.addAll(entries)
        vectorStore.save()
        return@withContext entries.size
    }

    suspend fun ask(question: String, topK: Int = 5): String = withContext(Dispatchers.Default) {
        checkNotNull(smolLM) { "SmolLM must be initialized and loaded with a model before calling ask()" }
        val contextText = contextFor(question, topK)
        if (contextText.isBlank()) {
            Log.w(TAG, "No retrieval hits; vector store empty or no similar content")
            return@withContext "No relevant context found in the indexed documents. If your PDF is a scanned image, text extraction may be empty (no OCR). Try a text-based PDF."
        }
        val prompt = buildPrompt(contextText, question)
        // Provide a strong system instruction for each query (recommended when storeChats=false)
        smolLM.addSystemPrompt(SYSTEM_PROMPT)
        smolLM.addUserMessage(prompt)
        val answer = smolLM.getResponse("")
        return@withContext answer.trim()
    }
    suspend fun contextFor(question: String, topK: Int = 5): String = withContext(Dispatchers.Default) {
        val hitsWithScores = retrieve(question, topK)
        if (hitsWithScores.isEmpty()) {
            lastContext = ""
            return@withContext ""
        }
        val ctx = buildContextFromHits(hitsWithScores)
        lastContext = ctx
        return@withContext ctx
    }

    fun getLastContext(): String = lastContext


    private fun buildPrompt(context: String, query: String): String {
        return """
            Context (use only this):
            $context

            Task:
            Answer the question strictly and only from the context above. If the context is insufficient, say "I don't know".

            Question:
            $query
        """.trimIndent()
    }

    private fun buildContextFromHits(hitsWithScores: List<Pair<VectorEntry, Float>>): String {
        // Filter weak matches and truncate to avoid overlong prompts
        val minScore = 0.10f
        val sb = StringBuilder()
        var totalChars = 0
        val maxChars = 3000
        for ((entry, score) in hitsWithScores) {
            if (score < minScore) continue
            val piece = entry.text.trim()
            if (piece.isEmpty()) continue
            val header = "[score=${"%.3f".format(score)}]\n"
            val toAdd = (header + piece + "\n\n---\n\n")
            if (totalChars + toAdd.length > maxChars) break
            sb.append(toAdd)
            totalChars += toAdd.length
        }
        if (sb.isNotEmpty()) return sb.toString()
        // Fallback: include top-1 chunk even if below threshold
        if (hitsWithScores.isNotEmpty()) {
            val (entry, score) = hitsWithScores.first()
            val piece = entry.text.trim()
            if (piece.isNotEmpty()) {
                val header = "[score=${"%.3f".format(score)}]\n"
                return (header + piece.take(1500))
            }
        }
        return ""
    }

    suspend fun retrieve(question: String, topK: Int = 5): List<Pair<VectorEntry, Float>> = withContext(Dispatchers.Default) {
        val qEmb = embeddingProvider.encode(question)
        vectorStore.topKWithScores(qEmb, topK)
    }

    suspend fun retrievalPreview(question: String, topK: Int = 5): String = withContext(Dispatchers.Default) {
        val hits = retrieve(question, topK)
        if (hits.isEmpty()) return@withContext "(no hits)"
        hits.joinToString("\n\n") { (e, s) -> "score=${"%.3f".format(s)}\n" + e.text.take(300) }
    }

    companion object {
        private const val TAG = "RAGEngine"
        private const val SYSTEM_PROMPT = "You are a question answering assistant. Use only the provided context to answer. If the context does not contain the answer, say 'I don't know'."
    }
}
