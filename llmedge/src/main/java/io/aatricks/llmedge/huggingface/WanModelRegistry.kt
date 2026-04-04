package io.aatricks.llmedge.huggingface

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.InputStream

@Serializable
data class WanModelEntry(
    val modelId: String,
    val filename: String,
    val architecture: String? = null,
    val modelType: String? = null,
    val mobileSupported: Boolean = true,
    val recommendedMinRAM: String? = null,
    val description: String? = null,
    val quantization: String? = null,
    val t5ModelId: String? = null,
    val t5Filename: String? = null,
    val vaeFilename: String? = null,
    val sizeBytes: Long? = null,
)

object WanModelRegistry {
    private const val ASSET_NAME = "wan-models/model-registry.json"
    private val json = Json { ignoreUnknownKeys = true }
    private val cacheLock = Any()
    private val cachedEntriesByAssets = mutableMapOf<Int, List<WanModelEntry>>()

    fun loadFromAssets(context: Context): List<WanModelEntry> {
        val assetManager = context.assets
        val cacheKey = System.identityHashCode(assetManager)

        synchronized(cacheLock) {
            cachedEntriesByAssets[cacheKey]?.let { return it }
        }

        val input: InputStream = assetManager.open(ASSET_NAME)
        val text = input.bufferedReader().use { it.readText() }
        return json.decodeFromString<List<WanModelEntry>>(text).also { loaded ->
            synchronized(cacheLock) {
                cachedEntriesByAssets[cacheKey] = loaded
            }
        }
    }

    fun findById(context: Context, modelId: String): WanModelEntry? =
        loadFromAssets(context).firstOrNull { it.modelId.equals(modelId, ignoreCase = true) }

    fun findByModelIdPrefix(context: Context, modelIdPrefix: String): WanModelEntry? =
        loadFromAssets(context).firstOrNull { it.modelId.startsWith(modelIdPrefix, ignoreCase = true) }
}
