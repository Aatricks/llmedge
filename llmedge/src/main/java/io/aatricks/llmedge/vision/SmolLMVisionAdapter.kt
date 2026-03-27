/*
 * Copyright (C) 2024 Aatricks
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

package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.text.runtime.SmolLM
import io.aatricks.llmedge.core.AndroidLogAdapter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Adapter that exposes SmolLM as a VisionModelAnalyzer implementation.
 *
 * This adapter currently implements a compatibility layer: it loads a regular
 * text model and, if prepared embeddings are present, replays them into the
 * model's KV cache so the model can answer image-grounded prompts.
 *
 * @property context Android context for file operations
 * @property smolLM The SmolLM instance to adapt
 */
class SmolLMVisionAdapter(
    private val context: Context,
    private val smolLM: SmolLM
) : VisionModelAnalyzer {
    
    companion object {
        private const val TAG = "SmolLMVision"
        private const val VISION_GENERATION_BATCH_SIZE = 1
    }
    
    // These will be set when loading a vision model
    private var modelPath: String? = null
    private var mmprojPath: String? = null
    private var hasVisionSupport = false
    
    /**
     * Load a vision-capable model with mmproj support.
     * 
     * @param modelPath Path to the vision-capable GGUF model (e.g., LLaVA)
     * @param mmprojPath Path to the mmproj file for CLIP vision encoder
     * @param params Optional inference parameters
     */
    suspend fun loadVisionModel(
        modelPath: String,
        mmprojPath: String? = null,
        params: SmolLM.InferenceParams = SmolLM.InferenceParams()
    ) = withContext(Dispatchers.IO) {
        val capabilityMessage = VisionPromptSupport.unsupportedReason(modelPath, mmprojPath)
        if (!checkVisionSupport(modelPath, mmprojPath)) {
            hasVisionSupport = false
            this@SmolLMVisionAdapter.modelPath = modelPath
            this@SmolLMVisionAdapter.mmprojPath = mmprojPath
            AndroidLogAdapter.w(TAG, capabilityMessage)
            throw UnsupportedOperationException(capabilityMessage)
        }

        try {
            this@SmolLMVisionAdapter.modelPath = modelPath
            this@SmolLMVisionAdapter.mmprojPath = mmprojPath

            smolLM.load(modelPath, params)
            hasVisionSupport = true

            AndroidLogAdapter.d(TAG, "Loaded vision model: $modelPath")
            if (mmprojPath != null) {
                AndroidLogAdapter.d(TAG, "With mmproj: $mmprojPath")
            }
        } catch (e: Exception) {
            hasVisionSupport = false
            AndroidLogAdapter.e(TAG, "Failed to load vision model", e)
            throw e
        }
    }
    
    override suspend fun analyze(
        image: ImageSource,
        prompt: String,
        params: VisionParams
    ): VisionResult = withContext(Dispatchers.IO) {
        if (!hasVisionCapabilities()) {
            throw UnsupportedOperationException(
                VisionPromptSupport.unsupportedReason(
                    modelPath = modelPath ?: "unknown",
                    projectorPath = mmprojPath,
                ),
            )
        }

        val startTime = System.currentTimeMillis()

        try {
            val imageFile = ImageUtils.imageToFile(context, image, "vision_input.jpg")
            val embdFile = File(imageFile.parentFile, "${imageFile.nameWithoutExtension}.bin")
            val metaFile = File(embdFile.absolutePath + ".meta.json")
            val usingPreparedEmbeddings = imageFile.extension.equals("bin", ignoreCase = true)

            if (!usingPreparedEmbeddings) {
                val projectorPathValue = mmprojPath
                if (!projectorPathValue.isNullOrBlank()) {
                    val projector = Projector()
                    try {
                        projector.init(projectorPathValue, smolLM.getNativeModelPointer())
                        check(projector.isReady()) {
                            "Native projector initialization failed for ${File(projectorPathValue).name}."
                        }
                        val primed = smolLM.primeImageBuffer(projector.nativeHandle(), imageFile.readBytes(), params.nBatch ?: 1)
                        check(primed) {
                            "The current runtime could not prime multimodal context for ${File(projectorPathValue).name}."
                        }

                        val finalPrompt =
                            params.systemPrompt
                                ?.takeUnless(String::isBlank)
                                ?.let { systemPrompt -> "$systemPrompt\n\n$prompt" }
                                ?: prompt
                        val response =
                            smolLM.getResponse(
                                query = finalPrompt,
                                maxTokens = params.maxTokens,
                                batchSize = VISION_GENERATION_BATCH_SIZE,
                            )
                        val duration = System.currentTimeMillis() - startTime

                        val tokensIn = estimateTokens(finalPrompt)
                        val tokensOut = estimateTokens(response)

                        return@withContext VisionResult(
                            text = response,
                            durationMs = duration,
                            modelId = getModelId(),
                            tokensIn = tokensIn,
                            tokensOut = tokensOut,
                        )
                    } finally {
                        projector.close()
                    }
                }
            }

            val visionPrompt = prompt
            if (!embdFile.exists() || !metaFile.exists()) {
                throw IllegalStateException(
                    "Prepared multimodal embeddings are missing. Ensure the projector mmproj file matches the model and native projector support is available.",
                )
            }

            AndroidLogAdapter.d(TAG, "Decoding prepared embeddings from ${embdFile.absolutePath}")
            val decodeOk =
                smolLM.decodePreparedEmbeddings(
                    embdFile.absolutePath,
                    metaFile.absolutePath,
                    params.nBatch ?: 1,
                )
            if (!decodeOk) {
                throw IllegalStateException(
                    "The current runtime could not decode prepared image embeddings for this model/projector combination.",
                )
            }

            val finalPrompt =
                params.systemPrompt
                    ?.takeUnless(String::isBlank)
                    ?.let { systemPrompt -> "$systemPrompt\n\n$visionPrompt" }
                    ?: visionPrompt
            val response =
                smolLM.getResponse(
                    query = finalPrompt,
                    maxTokens = params.maxTokens,
                    batchSize = VISION_GENERATION_BATCH_SIZE,
                )
            val duration = System.currentTimeMillis() - startTime

            val tokensIn = estimateTokens(finalPrompt)
            val tokensOut = estimateTokens(response)

            VisionResult(
                text = response,
                durationMs = duration,
                modelId = getModelId(),
                tokensIn = tokensIn,
                tokensOut = tokensOut
            )
        } catch (e: Exception) {
            AndroidLogAdapter.e(TAG, "Vision analysis failed", e)
            when (e) {
                is IllegalStateException,
                is UnsupportedOperationException -> throw e
                else -> throw IllegalStateException("Vision analysis failed: ${e.message}", e)
            }
        }
    }
    
    /**
     * Decode pre-computed vision embeddings directly from memory into the KV cache, avoiding
     * all file I/O. Returns true on success.
     */
    fun decodeEmbeddingsBuffer(embeddings: VisionEmbeddings, nBatch: Int = 1): Boolean {
        if (!hasVisionCapabilities()) return false
        return try {
            smolLM.decodeEmbeddingsBuffer(embeddings, nBatch)
        } catch (e: UnsatisfiedLinkError) {
            AndroidLogAdapter.w(TAG, "decodeEmbeddingsBuffer not available: ${e.message}")
            false
        }
    }

    override fun hasVisionCapabilities(): Boolean {
        // This will check native vision support when implemented
        return hasVisionSupport
    }
    
    override fun getModelId(): String {
        return modelPath?.let { File(it).nameWithoutExtension } ?: "unknown"
    }
    
    /**
     * Check if a model file supports vision capabilities.
     * This is a placeholder that will read actual model metadata.
     */
    private fun checkVisionSupport(modelPath: String): Boolean {
        return VisionPromptSupport.appearsVisionCapable(modelPath)
    }

    private fun checkVisionSupport(modelPath: String, mmprojPath: String?): Boolean {
        return VisionPromptSupport.isReadyForMultimodalInference(modelPath, mmprojPath)
    }
    
    /**
     * Format the prompt for vision models.
     * Different models may require different formats.
     */
    private fun formatVisionPrompt(prompt: String, imageFile: File): String {
        return VisionPromptSupport.formatVisionPrompt(prompt, imageFile)
    }
    
    /**
     * Estimate token count for a string.
     * Rough approximation: ~4 characters per token.
     */
    private fun estimateTokens(text: String): Int {
        return VisionPromptSupport.estimateTokens(text)
    }
    
    /**
     * Release resources.
     */
    fun close() {
        // Do not close smolLM as the caller owns the underlying model lifecycle.
        // smolLM.close()
        hasVisionSupport = false
        modelPath = null
        mmprojPath = null
    }
}

/**
 * Extension function to check if a SmolLM instance can be used for vision.
 */
fun SmolLM.toVisionAdapter(context: Context): SmolLMVisionAdapter {
    return SmolLMVisionAdapter(context, this)
}
