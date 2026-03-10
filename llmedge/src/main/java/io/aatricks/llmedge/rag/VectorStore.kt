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

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.util.PriorityQueue
import kotlin.math.sqrt

data class VectorEntry(
    val id: String,
    val text: String,
    val embedding: FloatArray,
)

class InMemoryVectorStore(private val persistFile: File? = null) {
    private val entries = mutableListOf<VectorEntry>()
    private val idIndex = HashMap<String, Int>()
    private val cachedNorms = mutableListOf<Float>()
    private val gson = Gson()

    companion object {
        private const val BRUTE_FORCE_THRESHOLD = 500
    }

    // Partition index for sub-linear retrieval on large datasets
    private var partitionIndex: PartitionIndex? = null

    fun upsert(entry: VectorEntry) {
        val existingIdx = idIndex[entry.id]
        if (existingIdx != null) {
            entries[existingIdx] = entry
            cachedNorms[existingIdx] = norm(entry.embedding)
        } else {
            val idx = entries.size
            entries.add(entry)
            cachedNorms.add(norm(entry.embedding))
            idIndex[entry.id] = idx
        }
        invalidateIndex()
    }

    fun addAll(newEntries: List<VectorEntry>) {
        for (entry in newEntries) {
            val existingIdx = idIndex[entry.id]
            if (existingIdx != null) {
                entries[existingIdx] = entry
                cachedNorms[existingIdx] = norm(entry.embedding)
            } else {
                val idx = entries.size
                entries.add(entry)
                cachedNorms.add(norm(entry.embedding))
                idIndex[entry.id] = idx
            }
        }
        invalidateIndex()
    }

    fun isEmpty() = entries.isEmpty()

    fun topK(query: FloatArray, k: Int = 5): List<VectorEntry> =
        topKWithScores(query, k).map { it.first }

    fun topKWithScores(query: FloatArray, k: Int = 5): List<Pair<VectorEntry, Float>> {
        if (entries.isEmpty()) return emptyList()
        val effectiveK = minOf(k, entries.size)

        return if (entries.size >= BRUTE_FORCE_THRESHOLD) {
            indexedTopK(query, effectiveK)
        } else {
            bruteForceTopK(query, effectiveK)
        }
    }

    fun head(n: Int): List<VectorEntry> = entries.take(n)
    fun size(): Int = entries.size

    private fun bruteForceTopK(query: FloatArray, k: Int): List<Pair<VectorEntry, Float>> {
        val qnorm = norm(query)
        // Min-heap bounded to k elements — O(N log K) instead of O(N log N)
        val heap = PriorityQueue<Pair<Int, Float>>(k + 1, compareBy { it.second })
        for (i in entries.indices) {
            val score = cosine(query, qnorm, entries[i].embedding, cachedNorms[i])
            if (heap.size < k) {
                heap.add(i to score)
            } else if (score > heap.peek().second) {
                heap.poll()
                heap.add(i to score)
            }
        }
        return heap.sortedByDescending { it.second }.map { (idx, score) -> entries[idx] to score }
    }

    /**
     * Random partition tree index for sub-linear retrieval.
     * Partitions embeddings by random hyperplanes, then searches only relevant partitions.
     */
    private fun indexedTopK(query: FloatArray, k: Int): List<Pair<VectorEntry, Float>> {
        val index = partitionIndex ?: buildPartitionIndex().also { partitionIndex = it }

        // Collect candidate indices from the partition tree
        val candidates = index.getCandidates(query, k)

        // Score only the candidates
        val qnorm = norm(query)
        val heap = PriorityQueue<Pair<Int, Float>>(k + 1, compareBy { it.second })
        for (idx in candidates) {
            val score = cosine(query, qnorm, entries[idx].embedding, cachedNorms[idx])
            if (heap.size < k) {
                heap.add(idx to score)
            } else if (score > heap.peek().second) {
                heap.poll()
                heap.add(idx to score)
            }
        }
        return heap.sortedByDescending { it.second }.map { (idx, score) -> entries[idx] to score }
    }

    private fun buildPartitionIndex(): PartitionIndex {
        val indices = entries.indices.toList()
        val dim = if (entries.isNotEmpty()) entries[0].embedding.size else 0
        return PartitionIndex.build(entries, dim, indices)
    }

    private fun invalidateIndex() {
        partitionIndex = null
    }

    private fun norm(x: FloatArray): Float {
        var s = 0f
        for (v in x) s += v * v
        return sqrt(s)
    }

    private fun cosine(a: FloatArray, an: Float, b: FloatArray, bn: Float): Float {
        var dot = 0f
        val n = minOf(a.size, b.size)
        for (i in 0 until n) {
            dot += a[i] * b[i]
        }
        val denom = an * bn
        return if (denom == 0f) 0f else dot / denom
    }

    fun save() {
        persistFile ?: return
        val serializable = entries.map { e -> SerializableEntry(e.id, e.text, e.embedding.toList()) }
        persistFile.parentFile?.mkdirs()
        persistFile.writeText(gson.toJson(serializable))
    }

    fun load() {
        persistFile ?: return
        if (!persistFile.exists()) return
        val type = object : TypeToken<List<SerializableEntry>>() {}.type
        val list: List<SerializableEntry> = gson.fromJson(persistFile.readText(), type) ?: return
        entries.clear()
        idIndex.clear()
        cachedNorms.clear()
        invalidateIndex()
        for ((i, item) in list.withIndex()) {
            val entry = VectorEntry(item.id, item.text, item.embedding.toFloatArray())
            entries.add(entry)
            cachedNorms.add(norm(entry.embedding))
            idIndex[entry.id] = i
        }
    }

    private data class SerializableEntry(
        val id: String,
        val text: String,
        val embedding: List<Float>,
    )

    /**
     * Random partition tree for approximate nearest neighbor search.
     * Each internal node splits entries by a random hyperplane; leaves hold entry indices.
     * Multiple trees are searched and results are merged for better recall.
     */
    internal class PartitionIndex private constructor(private val trees: List<Node>) {

        companion object {
            private const val LEAF_SIZE = 50
            private const val NUM_TREES = 3

            fun build(entries: List<VectorEntry>, dim: Int, indices: List<Int>): PartitionIndex {
                val trees = (0 until NUM_TREES).map { seed ->
                    buildTree(entries, dim, indices, depth = 0, seed = seed * 7919)
                }
                return PartitionIndex(trees)
            }

            private fun buildTree(
                entries: List<VectorEntry>,
                dim: Int,
                indices: List<Int>,
                depth: Int,
                seed: Int
            ): Node {
                if (indices.size <= LEAF_SIZE || dim == 0) {
                    return Node.Leaf(indices)
                }

                // Pick two random pivot points
                val rng = java.util.Random((seed.toLong() * 31 + depth.toLong()) * 37 + indices.size.toLong())
                val pivotA = indices[rng.nextInt(indices.size)]
                val pivotB = indices[rng.nextInt(indices.size)]
                if (pivotA == pivotB) return Node.Leaf(indices)

                // Compute hyperplane normal as difference of pivots
                val embA = entries[pivotA].embedding
                val embB = entries[pivotB].embedding
                val hyperplane = FloatArray(dim)
                var midpoint = 0f
                for (i in 0 until minOf(dim, embA.size, embB.size)) {
                    hyperplane[i] = embA[i] - embB[i]
                    midpoint += hyperplane[i] * (embA[i] + embB[i]) / 2f
                }

                val left = mutableListOf<Int>()
                val right = mutableListOf<Int>()
                for (idx in indices) {
                    var projection = 0f
                    val emb = entries[idx].embedding
                    for (i in 0 until minOf(dim, emb.size)) {
                        projection += hyperplane[i] * emb[i]
                    }
                    if (projection <= midpoint) left.add(idx) else right.add(idx)
                }

                // Prevent degenerate splits
                if (left.isEmpty() || right.isEmpty()) return Node.Leaf(indices)

                return Node.Internal(
                    hyperplane = hyperplane,
                    threshold = midpoint,
                    left = buildTree(entries, dim, left, depth + 1, seed),
                    right = buildTree(entries, dim, right, depth + 1, seed),
                )
            }
        }

        fun getCandidates(query: FloatArray, k: Int): Set<Int> {
            val candidates = mutableSetOf<Int>()
            for (tree in trees) {
                tree.search(query, k, candidates)
            }
            return candidates
        }

        internal sealed class Node {
            abstract fun search(query: FloatArray, k: Int, candidates: MutableSet<Int>)

            class Leaf(private val indices: List<Int>) : Node() {
                override fun search(query: FloatArray, k: Int, candidates: MutableSet<Int>) {
                    candidates.addAll(indices)
                }
            }

            class Internal(
                private val hyperplane: FloatArray,
                private val threshold: Float,
                private val left: Node,
                private val right: Node,
            ) : Node() {
                override fun search(query: FloatArray, k: Int, candidates: MutableSet<Int>) {
                    var projection = 0f
                    for (i in 0 until minOf(hyperplane.size, query.size)) {
                        projection += hyperplane[i] * query[i]
                    }
                    // Search the primary side first, then the other side for better recall
                    if (projection <= threshold) {
                        left.search(query, k, candidates)
                        if (candidates.size < k * 3) {
                            right.search(query, k, candidates)
                        }
                    } else {
                        right.search(query, k, candidates)
                        if (candidates.size < k * 3) {
                            left.search(query, k, candidates)
                        }
                    }
                }
            }
        }
    }
}
