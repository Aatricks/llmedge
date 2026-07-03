package io.aatricks.llmedge.model

import io.aatricks.llmedge.runtime.GGUFReader
import java.io.File
import kotlinx.coroutines.runBlocking

internal data class GgufModelMetadata(
    val architecture: String?,
    val modelName: String?,
    val parameterCount: String?,
)

internal object GgufModelMetadataSupport {
    private val lock = Any()
    private val metadataCache = mutableMapOf<String, GgufModelMetadata?>()

    fun inspect(modelPath: String): GgufModelMetadata? {
        if (!modelPath.endsWith(".gguf", ignoreCase = true)) {
            return null
        }
        val file = File(modelPath)
        if (!file.exists() || !file.isFile || !file.canRead()) {
            return null
        }
        synchronized(lock) {
            if (metadataCache.containsKey(file.absolutePath)) {
                return metadataCache[file.absolutePath]
            }
        }
        val metadata = loadMetadata(file)
        if (metadata != null) {
            synchronized(lock) {
                metadataCache[file.absolutePath] = metadata
            }
        }
        return metadata
    }

    private fun loadMetadata(file: File): GgufModelMetadata? =
        runCatching {
            val reader = GGUFReader()
            try {
                runBlocking {
                    reader.load(file.absolutePath)
                }
                GgufModelMetadata(
                    architecture = reader.getArchitecture(),
                    modelName = reader.getModelName(),
                    parameterCount = reader.getParameterCount(),
                ).takeIf { metadata ->
                    metadata.architecture != null || metadata.modelName != null || metadata.parameterCount != null
                }
            } finally {
                reader.close()
            }
        }.getOrNull()
}
