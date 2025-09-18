package io.shubham0204.smollm.rag

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlin.math.sqrt
import java.io.File

data class VectorEntry(
    val id: String,
    val text: String,
    val embedding: FloatArray,
)

class InMemoryVectorStore(private val persistFile: File? = null) {
    private val entries = mutableListOf<VectorEntry>()
    private val gson = Gson()

    fun upsert(entry: VectorEntry) {
        val idx = entries.indexOfFirst { it.id == entry.id }
        if (idx >= 0) entries[idx] = entry else entries.add(entry)
    }

    fun addAll(newEntries: List<VectorEntry>) { newEntries.forEach { upsert(it) } }

    fun isEmpty() = entries.isEmpty()

    fun topK(query: FloatArray, k: Int = 5): List<VectorEntry> {
        if (entries.isEmpty()) return emptyList()
        val qnorm = norm(query)
        return entries
            .asSequence()
            .map { e -> e to cosine(query, qnorm, e.embedding) }
            .sortedByDescending { it.second }
            .take(k)
            .map { it.first }
            .toList()
    }

    fun topKWithScores(query: FloatArray, k: Int = 5): List<Pair<VectorEntry, Float>> {
        if (entries.isEmpty()) return emptyList()
        val qnorm = norm(query)
        return entries
            .asSequence()
            .map { e -> e to cosine(query, qnorm, e.embedding) }
            .sortedByDescending { it.second }
            .take(k)
            .toList()
    }

    fun head(n: Int): List<VectorEntry> = entries.take(n)
    fun size(): Int = entries.size

    private fun norm(x: FloatArray): Float {
        var s = 0f
        for (v in x) s += v * v
        return sqrt(s)
    }

    private fun cosine(a: FloatArray, an: Float, b: FloatArray): Float {
        var dot = 0f
        var bn = 0f
        val n = minOf(a.size, b.size)
        for (i in 0 until n) {
            dot += a[i] * b[i]
            bn += b[i] * b[i]
        }
        val denom = an * sqrt(bn)
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
        entries.addAll(list.map { VectorEntry(it.id, it.text, it.embedding.toFloatArray()) })
    }

    private data class SerializableEntry(
        val id: String,
        val text: String,
        val embedding: List<Float>,
    )
}
