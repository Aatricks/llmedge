package io.shubham0204.smollm.rag

/**
 * Produces overlapping chunks from a long text using whitespace boundaries.
 */
class TextSplitter(
    private val chunkSize: Int = 400,
    private val chunkOverlap: Int = 80,
) {
    init {
        require(chunkSize > 0) { "chunkSize must be > 0" }
        require(chunkOverlap in 0 until chunkSize) { "chunkOverlap must be in [0, chunkSize)" }
    }

    fun split(text: String): List<String> {
        if (text.isBlank()) return emptyList()
        val tokens = text.split(Regex("\\s+"))
        if (tokens.isEmpty()) return emptyList()
        val result = mutableListOf<String>()
        var start = 0
        val stride = (chunkSize - chunkOverlap).coerceAtLeast(1)
        while (start < tokens.size) {
            val end = (start + chunkSize).coerceAtMost(tokens.size)
            result.add(tokens.subList(start, end).joinToString(" "))
            if (end == tokens.size) break
            start += stride
        }
        return result
    }
}
