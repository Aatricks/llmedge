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

/**
 * Produces overlapping chunks from a long text using whitespace boundaries.
 */
class TextSplitter(
    private val chunkSize: Int = 80,
    private val chunkOverlap: Int = 16,
) {
    init {
        require(chunkSize > 0) { "chunkSize must be > 0" }
        require(chunkOverlap in 0 until chunkSize) { "chunkOverlap must be in [0, chunkSize)" }
    }

    fun split(text: String): List<String> {
        if (text.isBlank()) return emptyList()
        val tokens = text.split(Regex("\\s+"))
        if (tokens.isEmpty()) return emptyList()

        val averageWordLength = 5
        val limit = 4 * averageWordLength
        val hasNoUsableWhitespace = tokens.any { it.length > limit }

        if (hasNoUsableWhitespace) {
            val codePoints = mutableListOf<String>()
            var i = 0
            while (i < text.length) {
                val codePoint = text.codePointAt(i)
                val charCount = Character.charCount(codePoint)
                codePoints.add(text.substring(i, i + charCount))
                i += charCount
            }
            val result = mutableListOf<String>()
            var start = 0
            val stride = (chunkSize - chunkOverlap).coerceAtLeast(1)
            while (start < codePoints.size) {
                val end = (start + chunkSize).coerceAtMost(codePoints.size)
                result.add(codePoints.subList(start, end).joinToString(""))
                if (end == codePoints.size) break
                start += stride
            }
            return result
        }

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
