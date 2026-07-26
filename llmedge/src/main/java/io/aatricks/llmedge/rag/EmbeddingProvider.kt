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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import com.ml.shubham0204.sentence_embeddings.SentenceEmbedding
import java.io.File
import java.io.InputStream
import java.net.URI
import java.security.MessageDigest
import kotlinx.coroutines.runBlocking

data class EmbeddingConfig(
    val modelAssetPath: String = "embeddings/all-minilm-l6-v2/model.onnx",
    val tokenizerAssetPath: String = "embeddings/all-minilm-l6-v2/tokenizer.json",
    val useTokenTypeIds: Boolean = false,
    val outputTensorName: String = "sentence_embedding",
    val useFP16: Boolean = false,
    val useXNNPack: Boolean = true,
) {
    companion object {
        @JvmStatic
        @JvmOverloads
        fun fromFiles(
            modelFile: File,
            tokenizerFile: File,
            useTokenTypeIds: Boolean = false,
            outputTensorName: String = "sentence_embedding",
            useFP16: Boolean = false,
            useXNNPack: Boolean = true,
        ): EmbeddingConfig =
            EmbeddingConfig(
                modelAssetPath = modelFile.toURI().toASCIIString(),
                tokenizerAssetPath = tokenizerFile.toURI().toASCIIString(),
                useTokenTypeIds = useTokenTypeIds,
                outputTensorName = outputTensorName,
                useFP16 = useFP16,
                useXNNPack = useXNNPack,
            )
    }
}

internal sealed interface EmbeddingFileSource {
    fun materialize(
        cachedFile: File,
        assetOpener: (String) -> InputStream,
    ): File

    data class Asset(private val path: String) : EmbeddingFileSource {
        override fun materialize(
            cachedFile: File,
            assetOpener: (String) -> InputStream,
        ): File {
            if (cachedFile.isFile) return cachedFile
            cachedFile.parentFile?.mkdirs()
            val tmpFile = File(cachedFile.parent, "${cachedFile.name}.${System.nanoTime()}.tmp")
            try {
                assetOpener(path).use { input ->
                    tmpFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                if (!tmpFile.renameTo(cachedFile)) {
                    tmpFile.copyTo(cachedFile, overwrite = true)
                    tmpFile.delete()
                }
                return cachedFile
            } catch (t: Throwable) {
                tmpFile.delete()
                throw t
            }
        }
    }

    data class LocalFile(private val file: File) : EmbeddingFileSource {
        override fun materialize(
            cachedFile: File,
            assetOpener: (String) -> InputStream,
        ): File {
            require(file.isFile && file.canRead()) {
                "Embedding file is not readable: ${file.absolutePath}"
            }
            return file
        }
    }

    companion object {
        fun parse(path: String): EmbeddingFileSource =
            if (path.startsWith("file:", ignoreCase = true)) {
                LocalFile(File(URI(path)))
            } else {
                Asset(path)
            }
    }
}

class EmbeddingProvider(private val context: Context, private var config: EmbeddingConfig = EmbeddingConfig()) {
    private val se = SentenceEmbedding()
    private var initialized = false
    private var modelFilePath: String? = null
    private var tokenizerBytesCache: ByteArray? = null
    private var closed = false

    // Serializes init/encode: the token_type_ids fallback mutates `config` and
    // re-inits `se`, which must not happen while another encode is in flight
    // (concurrent indexing vs. querying share this provider).
    private val sessionMutex = Mutex()

    private fun getUniqueFileName(assetPath: String): String {
        val basename = File(assetPath).name
        val hash = try {
            val md = MessageDigest.getInstance("MD5")
            val bytes = md.digest(assetPath.toByteArray(Charsets.UTF_8))
            bytes.joinToString("") { "%02x".format(it) }.take(8)
        } catch (e: Exception) {
            assetPath.hashCode().toString(16)
        }
        return "${hash}_$basename"
    }

    suspend fun init() = sessionMutex.withLock {
        check(!closed) { "EmbeddingProvider is closed" }
        withContext(Dispatchers.IO) {
            if (initialized) return@withContext
            val modelsDir = File(context.filesDir, "embedding_models")
            val modelFile =
                EmbeddingFileSource.parse(config.modelAssetPath).materialize(
                    File(modelsDir, getUniqueFileName(config.modelAssetPath)),
                    context.assets::open,
                )
            val tokenizerFile =
                EmbeddingFileSource.parse(config.tokenizerAssetPath).materialize(
                    File(modelsDir, getUniqueFileName(config.tokenizerAssetPath)),
                    context.assets::open,
                )

            val tokenizerBytes = tokenizerFile.readBytes()
            modelFilePath = modelFile.absolutePath
            tokenizerBytesCache = tokenizerBytes
            initInternal()
            // Probe to auto-adapt configuration for models requiring token_type_ids
            try {
                se.encode("__init_probe__")
            } catch (t: Throwable) {
                val msg = t.message ?: ""
                val needsTokenTypeIds = msg.contains("Missing Input: token_type_ids", ignoreCase = true)
                if (needsTokenTypeIds && !config.useTokenTypeIds) {
                    config = config.copy(useTokenTypeIds = true, outputTensorName = "last_hidden_state")
                    initInternal()
                } else {
                    // ignore; actual encode() will still catch and re-adapt if needed
                }
            }
            initialized = true
        }
    }

    suspend fun encode(text: String): FloatArray = sessionMutex.withLock {
        check(!closed) { "EmbeddingProvider is closed" }
        withContext(Dispatchers.IO) {
            check(initialized) { "EmbeddingProvider.init() must be called first" }
            try {
                se.encode(text)
            } catch (t: Throwable) {
                // Auto-fallback for models that require token_type_ids (e.g., bge-small-en-v1.5)
                val msg = t.message ?: ""
                val needsTokenTypeIds = msg.contains("Missing Input: token_type_ids", ignoreCase = true)
                if (needsTokenTypeIds && !config.useTokenTypeIds) {
                    // Re-init with token type ids and common output tensor for such models
                    config = config.copy(useTokenTypeIds = true, outputTensorName = "last_hidden_state")
                    initInternal()
                    se.encode(text)
                } else {
                    throw t
                }
            }
        }
    }

    private suspend fun initInternal() {
        val model = requireNotNull(modelFilePath) { "Model path missing" }
        val tokenizerBytes = requireNotNull(tokenizerBytesCache) { "Tokenizer bytes missing" }
        se.init(
            modelFilepath = model,
            tokenizerBytes = tokenizerBytes,
            useTokenTypeIds = config.useTokenTypeIds,
            outputTensorName = config.outputTensorName,
            useFP16 = config.useFP16,
            useXNNPack = config.useXNNPack,
            normalizeEmbeddings = true,
        )
    }

    fun close() = runBlocking {
        sessionMutex.withLock {
            if (!closed) {
                if (initialized) {
                    se.close()
                }
                closed = true
            }
        }
    }
}
