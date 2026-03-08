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

package io.aatricks.llmedge.vision.ocr

import android.content.Context
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.vision.*
import kotlinx.coroutines.tasks.await

/**
 * OCR engine implementation using Google ML Kit Text Recognition.
 *
 * @property context Android context for accessing resources.
 */
class MlKitOcrEngine(
    private val context: Context
) : OcrEngine {
    companion object {
        private const val LOG_TAG = "MlKitOcrEngine"
    }
    
    private var textRecognizer: TextRecognizer? = null
    
    /**
     * Initialize the ML Kit text recognizer lazily.
     */
    private fun getOrCreateRecognizer(): TextRecognizer {
        return textRecognizer ?: TextRecognition.getClient(
            TextRecognizerOptions.DEFAULT_OPTIONS
        ).also {
            textRecognizer = it
        }
    }
    
    override suspend fun extractText(
        image: ImageSource,
        params: OcrParams
    ): OcrResult {
        val startTime = System.currentTimeMillis()
        AndroidLogAdapter.d(LOG_TAG, "extractText: start, params=$params, source=${image.javaClass.simpleName}")
        
        try {
            val bitmap = try {
                if (params.enhance) {
                    ImageUtils.preprocessImage(
                        context = context,
                        source = image,
                        enhance = true,
                    )
                } else {
                    ImageUtils.imageToBitmap(context, image)
                }
            } catch (e: Exception) {
                AndroidLogAdapter.e(LOG_TAG, "Failed to convert ImageSource to Bitmap", e)
                throw e
            }

            // Create InputImage for ML Kit
            val inputImage = try {
                InputImage.fromBitmap(bitmap, 0)
            } catch (e: Exception) {
                AndroidLogAdapter.e(LOG_TAG, "Failed to create InputImage", e)
                throw e
            }
            
            // Get recognizer
            val recognizer = getOrCreateRecognizer()
            
            // Process the image
            val result = try {
                recognizer.process(inputImage).await()
            } catch (e: Exception) {
                AndroidLogAdapter.e(LOG_TAG, "ML Kit recognizer.process failed", e)
                throw e
            }
            
            // Extract text from result
            val extractedText = buildString {
                for (block in result.textBlocks) {
                    appendLine(block.text)
                }
            }.trim()
            
            // Calculate confidence (ML Kit doesn't provide overall confidence, so we estimate)
            val confidence = if (extractedText.isNotEmpty()) {
                // Estimate confidence based on text length and structure
                val words = extractedText.split(Regex("\\s+"))
                val avgWordLength = if (words.isNotEmpty()) {
                    words.map { it.length }.average()
                } else {
                    0.0
                }
                
                // Simple heuristic: reasonable word length suggests better recognition
                when {
                    avgWordLength < 2 -> 0.3f
                    avgWordLength < 3 -> 0.5f
                    avgWordLength < 8 -> 0.8f
                    else -> 0.7f
                }
            } else {
                0.0f
            }
            
            val duration = System.currentTimeMillis() - startTime
            AndroidLogAdapter.d(LOG_TAG, "extractText: completed in ${duration}ms, textLength=${extractedText.length}")
            
            return OcrResult(
                text = extractedText,
                language = params.language,
                durationMs = duration,
                engine = getEngineName(),
                confidence = confidence
            )
            
        } catch (e: Exception) {
            val duration = System.currentTimeMillis() - startTime
            throw Exception("ML Kit OCR failed after ${duration}ms: ${e.message}", e)
        }
    }
    
    override suspend fun isAvailable(): Boolean {
        return try {
            // Check if ML Kit is available by trying to get a client
            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            true
        } catch (e: Exception) {
            false
        }
    }
    
    override fun getEngineName(): String = "mlkit"
    
    /**
     * Release resources when done.
     */
    fun close() {
        textRecognizer?.close()
        textRecognizer = null
    }
}