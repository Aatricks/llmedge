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
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
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
    private var storeDimension: Int = 0
    private var isLoaded = false

    // Guards entries/idIndex/cachedNorms/partitionIndex: nothing upstream serializes
    // indexing (addAll/upsert) against queries (topK) or save(), and ArrayList/HashMap
    // are unsafe under concurrent structural modification.
    private val lock = Any()
    private val saveLock = Any()

    companion object {
        private const val BRUTE_FORCE_THRESHOLD = 500

        // "LEVS" (llmedge vector store) + version header for the binary persist format.
        private val BINARY_MAGIC = byteArrayOf(0x4C, 0x45, 0x56, 0x53)
        private const val BINARY_VERSION = 2

        // Sanity bounds so a corrupt length field fails parsing instead of OOMing.
        private const val MAX_BLOB_BYTES = 16 * 1024 * 1024
        private const val MAX_EMBEDDING_DIM = 65536
    }

    // Partition index for sub-linear retrieval on large datasets
    private var partitionIndex: PartitionIndex? = null

    fun upsert(entry: VectorEntry) {
        val idBytes = entry.id.toByteArray(Charsets.UTF_8)
        val textBytes = entry.text.toByteArray(Charsets.UTF_8)
        require(idBytes.size <= MAX_BLOB_BYTES) { "Chunk ID is too large: ${idBytes.size} bytes (max $MAX_BLOB_BYTES)" }
        require(textBytes.size <= MAX_BLOB_BYTES) { "Chunk text is too large: ${textBytes.size} bytes (max $MAX_BLOB_BYTES)" }
        synchronized(lock) {
            if (storeDimension == 0 && entries.isEmpty()) {
                storeDimension = entry.embedding.size
            }
            if (storeDimension > 0) {
                require(entry.embedding.size == storeDimension) {
                    "Dimension mismatch: entry dimension ${entry.embedding.size} does not match store dimension $storeDimension. Please re-index your documents."
                }
            }
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
    }

    fun addAll(newEntries: List<VectorEntry>) {
        for (entry in newEntries) {
            val idBytes = entry.id.toByteArray(Charsets.UTF_8)
            val textBytes = entry.text.toByteArray(Charsets.UTF_8)
            require(idBytes.size <= MAX_BLOB_BYTES) { "Chunk ID is too large: ${idBytes.size} bytes (max $MAX_BLOB_BYTES)" }
            require(textBytes.size <= MAX_BLOB_BYTES) { "Chunk text is too large: ${textBytes.size} bytes (max $MAX_BLOB_BYTES)" }
        }
        synchronized(lock) {
            if (storeDimension == 0 && entries.isEmpty() && newEntries.isNotEmpty()) {
                storeDimension = newEntries[0].embedding.size
            }
            for (entry in newEntries) {
                if (storeDimension > 0) {
                    require(entry.embedding.size == storeDimension) {
                        "Dimension mismatch: entry dimension ${entry.embedding.size} does not match store dimension $storeDimension. Please re-index your documents."
                    }
                }
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
    }

    fun isEmpty() = synchronized(lock) { entries.isEmpty() }

    fun topK(query: FloatArray, k: Int = 5): List<VectorEntry> =
        topKWithScores(query, k).map { it.first }

    fun topKWithScores(query: FloatArray, k: Int = 5): List<Pair<VectorEntry, Float>> {
        synchronized(lock) {
            if (entries.isEmpty()) return emptyList()
            val effectiveK = minOf(k, entries.size)

            return if (entries.size >= BRUTE_FORCE_THRESHOLD) {
                indexedTopK(query, effectiveK)
            } else {
                bruteForceTopK(query, effectiveK)
            }
        }
    }

    fun head(n: Int): List<VectorEntry> = synchronized(lock) { entries.take(n) }
    fun size(): Int = synchronized(lock) { entries.size }

    private fun bruteForceTopK(query: FloatArray, k: Int): List<Pair<VectorEntry, Float>> {
        val qnorm = norm(query)
        // Min-heap bounded to k elements — O(N log K) instead of O(N log N)
        val heap = PriorityQueue<Pair<Int, Float>>(k + 1, compareBy { it.second })
        for (i in entries.indices) {
            val score = cosine(query, qnorm, entries[i].embedding, cachedNorms[i])
            if (heap.size < k) {
                heap.add(i to score)
            } else {
                val currentMin = heap.peek() ?: continue
                if (score <= currentMin.second) {
                    continue
                }
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
            } else {
                val currentMin = heap.peek() ?: continue
                if (score <= currentMin.second) {
                    continue
                }
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
        if (a.size != b.size) return 0f
        var dot = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
        }
        val denom = an * bn
        return if (denom == 0f) 0f else dot / denom
    }

    fun save() {
        val file = persistFile ?: return
        synchronized(saveLock) {
            val snapshot = synchronized(lock) { entries.toList() }
            val currentDim = synchronized(lock) { storeDimension }
            file.parentFile?.mkdirs()
            // Compact binary format (embeddings as raw floats instead of boxed JSON
            // numbers: ~4-6x smaller and much faster to (de)serialize at 10k+ chunks).
            // Written to a temp file and renamed so a crash mid-write can't leave a
            // truncated index behind.
            val tmp = File(file.parentFile, "${file.name}.${System.nanoTime()}.tmp")
            try {
                DataOutputStream(BufferedOutputStream(FileOutputStream(tmp))).use { out ->
                    out.write(BINARY_MAGIC)
                    out.writeInt(BINARY_VERSION)
                    out.writeInt(currentDim)
                    out.writeInt(snapshot.size)
                    for (e in snapshot) {
                        writeBlob(out, e.id.toByteArray(Charsets.UTF_8))
                        writeBlob(out, e.text.toByteArray(Charsets.UTF_8))
                        out.writeInt(e.embedding.size)
                        val bytes = ByteBuffer.allocate(e.embedding.size * 4)
                        bytes.asFloatBuffer().put(e.embedding)
                        out.write(bytes.array())
                    }
                }
                if (!tmp.renameTo(file)) {
                    tmp.copyTo(file, overwrite = true)
                    tmp.delete()
                }
            } catch (e: Throwable) {
                tmp.delete()
                throw e
            }
        }
    }

    fun load() {
        val file = persistFile ?: return
        if (!file.exists()) return
        synchronized(lock) {
            if (isLoaded) return
        }
        val loadedData: LoadedData =
            try {
                if (hasBinaryMagic(file)) {
                    loadBinary(file)
                } else {
                    val list = loadLegacyJson(file)
                    if (list == null) {
                        file.delete()
                        return
                    }
                    LoadedData(list, 0)
                }
            } catch (e: Throwable) {
                if (e is Error) throw e
                if (e is com.google.gson.JsonSyntaxException ||
                    e is IllegalArgumentException ||
                    e is IllegalStateException ||
                    e is java.io.EOFException ||
                    e is java.io.StreamCorruptedException) {
                    file.delete()
                }
                return
            }
        synchronized(lock) {
            if (isLoaded) return
            entries.clear()
            idIndex.clear()
            cachedNorms.clear()
            invalidateIndex()
            storeDimension = loadedData.dimension
            for ((i, entry) in loadedData.entries.withIndex()) {
                entries.add(entry)
                cachedNorms.add(norm(entry.embedding))
                idIndex[entry.id] = i
            }
            isLoaded = true
        }
    }

    private class LoadedData(val entries: List<VectorEntry>, val dimension: Int)

    private fun writeBlob(out: DataOutputStream, bytes: ByteArray) {
        require(bytes.size <= MAX_BLOB_BYTES) { "Chunk is too large: ${bytes.size} bytes (max $MAX_BLOB_BYTES)" }
        out.writeInt(bytes.size)
        out.write(bytes)
    }

    private fun hasBinaryMagic(file: File): Boolean {
        val head = ByteArray(BINARY_MAGIC.size)
        FileInputStream(file).use { input ->
            if (input.read(head) != head.size) return false
        }
        return head.contentEquals(BINARY_MAGIC)
    }

    private fun loadBinary(file: File): LoadedData {
        DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
            input.skipBytes(BINARY_MAGIC.size)
            val version = input.readInt()
            require(version == 1 || version == 2) { "Unsupported vector store version $version" }
            val dim = if (version == 2) input.readInt() else 0
            val count = input.readInt()
            require(count >= 0) { "Corrupt vector store entry count" }
            val result = ArrayList<VectorEntry>(count)
            repeat(count) {
                val id = String(readBlob(input), Charsets.UTF_8)
                val text = String(readBlob(input), Charsets.UTF_8)
                val entryDim = input.readInt()
                require(entryDim in 0..MAX_EMBEDDING_DIM) { "Corrupt embedding length $entryDim" }
                val raw = ByteArray(entryDim * 4)
                input.readFully(raw)
                val embedding = FloatArray(entryDim)
                ByteBuffer.wrap(raw).asFloatBuffer().get(embedding)
                result.add(VectorEntry(id, text, embedding))
            }
            return LoadedData(result, dim)
        }
    }

    private fun readBlob(input: DataInputStream): ByteArray {
        val size = input.readInt()
        require(size in 0..MAX_BLOB_BYTES) { "Corrupt blob length $size" }
        val bytes = ByteArray(size)
        input.readFully(bytes)
        return bytes
    }

    private fun loadLegacyJson(file: File): List<VectorEntry>? {
        val type = object : TypeToken<List<SerializableEntry>>() {}.type
        val list: List<SerializableEntry> =
            gson.fromJson<List<SerializableEntry>>(file.readText(), type) ?: return null
        return list.map { VectorEntry(it.id, it.text, it.embedding.toFloatArray()) }
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
