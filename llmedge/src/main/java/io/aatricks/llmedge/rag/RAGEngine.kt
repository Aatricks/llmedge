/*
 * Copyright (C) 2025 Aatricks
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

package io.aatricks.llmedge.rag

import android.content.Context
import android.net.Uri
import io.aatricks.llmedge.SmolLM
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
    private var systemPromptInjected = false

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
        ensureSystemPrompt()
        val prompt = buildPrompt(contextText, question)
        val answer = smolLM.getResponse(prompt)
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

    private fun ensureSystemPrompt() {
        if (!systemPromptInjected) {
            smolLM.addSystemPrompt(SYSTEM_PROMPT)
            systemPromptInjected = true
        }
    }

    private fun buildPrompt(context: String, query: String): String {
        return RAGPromptSupport.buildPrompt(context, query)
    }

    private fun buildContextFromHits(hitsWithScores: List<Pair<VectorEntry, Float>>): String {
        return RAGPromptSupport.buildContextFromHits(hitsWithScores)
    }

    suspend fun retrieve(question: String, topK: Int = 5): List<Pair<VectorEntry, Float>> = withContext(Dispatchers.Default) {
        val qEmb = embeddingProvider.encode(question)
        vectorStore.topKWithScores(qEmb, topK)
    }

    suspend fun retrievalPreview(question: String, topK: Int = 5): String = withContext(Dispatchers.Default) {
        val hits = retrieve(question, topK)
        RAGPromptSupport.formatRetrievalPreview(hits)
    }

    companion object {
        private const val TAG = "RAGEngine"
        private const val SYSTEM_PROMPT = "You are a question answering assistant. Use only the provided context to answer. If the context does not contain the answer, say 'I don't know'."
    }
}
