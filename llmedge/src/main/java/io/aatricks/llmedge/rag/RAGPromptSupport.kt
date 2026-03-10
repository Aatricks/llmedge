package io.aatricks.llmedge.rag

internal object RAGPromptSupport {
    fun buildPrompt(context: String, query: String): String =
        """
            Context (use only this):
            $context

            Task:
            Answer the question strictly and only from the context above. If the context is insufficient, say "I don't know".

            Question:
            $query
        """.trimIndent()

    fun buildContextFromHits(hitsWithScores: List<Pair<VectorEntry, Float>>): String {
        val minScore = 0.10f
        val builder = StringBuilder()
        var totalChars = 0
        val maxChars = 3000
        for ((entry, score) in hitsWithScores) {
            if (score < minScore) continue
            val piece = entry.text.trim()
            if (piece.isEmpty()) continue
            val toAdd = "[score=${"%.3f".format(score)}]\n$piece\n\n---\n\n"
            if (totalChars + toAdd.length > maxChars) break
            builder.append(toAdd)
            totalChars += toAdd.length
        }
        if (builder.isNotEmpty()) {
            return builder.toString()
        }

        if (hitsWithScores.isNotEmpty()) {
            val (entry, score) = hitsWithScores.first()
            val piece = entry.text.trim()
            if (piece.isNotEmpty()) {
                return "[score=${"%.3f".format(score)}]\n${piece.take(1500)}"
            }
        }
        return ""
    }

    fun formatRetrievalPreview(hits: List<Pair<VectorEntry, Float>>): String {
        if (hits.isEmpty()) {
            return "(no hits)"
        }
        return hits.joinToString("\n\n") { (entry, score) ->
            "score=${"%.3f".format(score)}\n${entry.text.take(300)}"
        }
    }
}