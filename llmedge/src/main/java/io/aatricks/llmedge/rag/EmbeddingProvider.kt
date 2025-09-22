/*
 * Copyright (C) 2025 Aatricks
 *
 * Licensed under the GNU General Public License v3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.gnu.org/licenses/gpl-3.0
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
import kotlinx.coroutines.withContext
import java.io.File
import com.ml.shubham0204.sentence_embeddings.SentenceEmbedding

data class EmbeddingConfig(
    val modelAssetPath: String = "embeddings/all-minilm-l6-v2/model.onnx",
    val tokenizerAssetPath: String = "embeddings/all-minilm-l6-v2/tokenizer.json",
    val useTokenTypeIds: Boolean = false,
    val outputTensorName: String = "sentence_embedding",
    val useFP16: Boolean = false,
    val useXNNPack: Boolean = false,
)

class EmbeddingProvider(private val context: Context, private var config: EmbeddingConfig = EmbeddingConfig()) {
    private val se = SentenceEmbedding()
    private var initialized = false
    private var modelFilePath: String? = null
    private var tokenizerBytesCache: ByteArray? = null

    private suspend fun copyAssetToFiles(assetPath: String, outFile: File) = withContext(Dispatchers.IO) {
        if (outFile.exists()) return@withContext
        outFile.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }

    suspend fun init() = withContext(Dispatchers.IO) {
        if (initialized) return@withContext
        val modelsDir = File(context.filesDir, "embedding_models")
        val modelFile = File(modelsDir, File(config.modelAssetPath).name)
        val tokenizerFile = File(modelsDir, File(config.tokenizerAssetPath).name)
        copyAssetToFiles(config.modelAssetPath, modelFile)
        copyAssetToFiles(config.tokenizerAssetPath, tokenizerFile)

        val tokenizerBytes = tokenizerFile.readBytes()
        modelFilePath = modelFile.absolutePath
        tokenizerBytesCache = tokenizerBytes
        initInternal()
        // Probe to auto-adapt configuration for models requiring token_type_ids
        try {
            withContext(Dispatchers.Default) { se.encode("__init_probe__") }
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

    suspend fun encode(text: String): FloatArray = withContext(Dispatchers.Default) {
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
                withContext(Dispatchers.IO) { initInternal() }
                se.encode(text)
            } else {
                throw t
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
}
