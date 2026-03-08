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
import android.util.Log
import io.aatricks.llmedge.SmolLM
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
        try {
            this@SmolLMVisionAdapter.modelPath = modelPath
            this@SmolLMVisionAdapter.mmprojPath = mmprojPath
            
            // For now, load as a regular model
            // When vision support is added to native code, this will call
            // the vision-specific loading function
            smolLM.load(modelPath, params)
            
            // Check if model supports vision (placeholder - will be determined from model metadata)
            hasVisionSupport = checkVisionSupport(modelPath)
            
            if (hasVisionSupport) {
                Log.d(TAG, "Loaded vision model: $modelPath")
                if (mmprojPath != null) {
                    Log.d(TAG, "With mmproj: $mmprojPath")
                }
            } else {
                Log.w(TAG, "Model does not appear to support vision: $modelPath")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vision model", e)
            throw e
        }
    }
    
    override suspend fun analyze(
        image: ImageSource,
        prompt: String,
        params: VisionParams
    ): VisionResult = withContext(Dispatchers.IO) {
        
        if (!hasVisionCapabilities()) {
            throw IllegalStateException("Vision capabilities not available. Load a vision model first.")
        }
        
        val startTime = System.currentTimeMillis()
        
        try {
            // Convert image to file for native processing
            val imageFile = ImageUtils.imageToFile(context, image, "vision_input.jpg")
            
            // Prepare the prompt for vision model
            // Different models may need different prompt formats
            val visionPrompt = formatVisionPrompt(prompt, imageFile)
            
            // If a prepared embedding exists (Projector.encodeImageToFile wrote .bin + .meta.json),
            // decode the embeddings into the current llama context without loading mmproj.
            val embdFile = File(imageFile.parentFile, "${imageFile.nameWithoutExtension}.bin")
            val metaFile = File(embdFile.absolutePath + ".meta.json")
            val response: String
            if (embdFile.exists() && metaFile.exists()) {
                Log.d(TAG, "Found prepared embeddings, decoding into model: ${embdFile.absolutePath}")
                val ok = smolLM.decodePreparedEmbeddings(embdFile.absolutePath, metaFile.absolutePath, params.nBatch ?: 1)
                if (!ok) {
                    Log.w(TAG, "decodePreparedEmbeddings failed, falling back to text-only prompt")
                    response = smolLM.getResponse(visionPrompt, params.maxTokens)
                } else {
                    // After embeddings are decoded into KV cache, call getResponse with the prompt
                    response = smolLM.getResponse(visionPrompt, params.maxTokens)
                }
            } else {
                // For now, use regular text generation as a placeholder
                // When vision support is added, this will call native vision inference
                response = smolLM.getResponse(visionPrompt, params.maxTokens)
            }
            
            val duration = System.currentTimeMillis() - startTime
            
            // Estimate tokens (placeholder - will be provided by native code)
            val tokensIn = estimateTokens(visionPrompt)
            val tokensOut = estimateTokens(response)
            
            VisionResult(
                text = response,
                durationMs = duration,
                modelId = getModelId(),
                tokensIn = tokensIn,
                tokensOut = tokensOut
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Vision analysis failed", e)
            throw Exception("Vision analysis failed: ${e.message}", e)
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
