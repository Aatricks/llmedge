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
import java.security.MessageDigest
import kotlinx.coroutines.runBlocking

data class EmbeddingConfig(
    val modelAssetPath: String = "embeddings/all-minilm-l6-v2/model.onnx",
    val tokenizerAssetPath: String = "embeddings/all-minilm-l6-v2/tokenizer.json",
    val useTokenTypeIds: Boolean = false,
    val outputTensorName: String = "sentence_embedding",
    val useFP16: Boolean = false,
    val useXNNPack: Boolean = true,
)

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

    private suspend fun copyAssetToFiles(assetPath: String, outFile: File) = withContext(Dispatchers.IO) {
        if (outFile.exists()) return@withContext
        outFile.parentFile?.mkdirs()
        val tmpFile = File(outFile.parent, "${outFile.name}.${System.nanoTime()}.tmp")
        try {
            context.assets.open(assetPath).use { input ->
                tmpFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            if (!tmpFile.renameTo(outFile)) {
                tmpFile.copyTo(outFile, overwrite = true)
                tmpFile.delete()
            }
        } catch (e: Throwable) {
            tmpFile.delete()
            throw e
        }
    }

    suspend fun init() = sessionMutex.withLock {
        check(!closed) { "EmbeddingProvider is closed" }
        withContext(Dispatchers.IO) {
            if (initialized) return@withContext
            val modelsDir = File(context.filesDir, "embedding_models")
            val modelFile = File(modelsDir, getUniqueFileName(config.modelAssetPath))
            val tokenizerFile = File(modelsDir, getUniqueFileName(config.tokenizerAssetPath))
            copyAssetToFiles(config.modelAssetPath, modelFile)
            copyAssetToFiles(config.tokenizerAssetPath, tokenizerFile)

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
