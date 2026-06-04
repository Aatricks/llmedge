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
import io.aatricks.llmedge.text.runtime.SmolLM
import java.io.File

/**
 * Minimal on-device RAG pipeline wiring:
 * - Document load (PDF only for now)
 * - Chunking via TextSplitter
 * - Embeddings via Sentence-Embeddings
 * - Vector search (cosine)
 * - Prompt building and answer via SmolLM
 *
 * Advanced API: most application code should prefer `LLMEdge.create(...).rag` and
 * `RAGClient.createSession()` so lifecycle and runtime ownership stay consistent with the rest of
 * the library.
 */
class RAGEngine(
    private val context: Context,
    private val smolLM: SmolLM,
    private val splitter: TextSplitter = TextSplitter(),
    embeddingConfig: EmbeddingConfig = EmbeddingConfig(),
) {
    private val embeddingProvider = EmbeddingProvider(context, embeddingConfig)
    @Volatile private var lastContext: String = ""
    private val vectorStore = InMemoryVectorStore(File(context.filesDir, "rag_store/index.json"))
    private val indexer = RAGIndexer(context, splitter, embeddingProvider, vectorStore)
    private val retriever = RAGRetriever(embeddingProvider, vectorStore)
    private val answerer = RAGAnswerer(smolLM)

    suspend fun init() {
        vectorStore.load()
        embeddingProvider.init()
    }

    suspend fun indexPdf(uri: Uri): Int = indexer.indexPdf(uri)

    suspend fun ask(question: String, topK: Int = 5): String {
        val contextText = contextFor(question, topK)
        lastContext = contextText
        return answerer.ask(question, contextText)
    }

    suspend fun contextFor(question: String, topK: Int = 5): String {
        val ctx = retriever.contextFor(question, topK)
        lastContext = ctx
        return ctx
    }

    fun getLastContext(): String = lastContext

    suspend fun retrieve(question: String, topK: Int = 5): List<Pair<VectorEntry, Float>> =
        retriever.retrieve(question, topK)

    suspend fun retrievalPreview(question: String, topK: Int = 5): String =
        retriever.retrievalPreview(question, topK)

    companion object {
        internal const val TAG = "RAGEngine"
        /** Hard cap on RAG answer length so small models stop before exhausting the context window. */
        internal const val MAX_ANSWER_TOKENS = 256
        internal const val SYSTEM_PROMPT =
            "You are a question answering assistant. Use only the provided context to answer. If the context does not contain the answer, say 'I don't know'."
    }
}
