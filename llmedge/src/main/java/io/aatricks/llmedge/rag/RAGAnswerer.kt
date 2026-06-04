package io.aatricks.llmedge.rag

import android.util.Log
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal class RAGAnswerer(
    private val smolLM: SmolLM,
) {
    private var systemPromptInjected = false

    suspend fun ask(
        question: String,
        contextText: String,
    ): String = withContext(Dispatchers.IO) {
        if (contextText.isBlank()) {
            Log.w(RAGEngine.TAG, "No retrieval hits; vector store empty or no similar content")
            return@withContext "No relevant context found in the indexed documents. If your PDF is a scanned image, text extraction may be empty (no OCR). Try a text-based PDF."
        }
        ensureSystemPrompt()
        val prompt = RAGPromptSupport.buildPrompt(contextText, question)
        // Cap the answer length. Without a cap (maxTokens = -1) small models that rarely emit EOS
        // keep generating until prompt+output fills the context window, which the native loop
        // surfaces as a hard "context size reached" IllegalStateException instead of an answer.
        smolLM.getResponse(prompt, maxTokens = RAGEngine.MAX_ANSWER_TOKENS).trim()
    }

    private fun ensureSystemPrompt() {
        if (!systemPromptInjected) {
            smolLM.addSystemPrompt(RAGEngine.SYSTEM_PROMPT)
            systemPromptInjected = true
        }
    }
}
