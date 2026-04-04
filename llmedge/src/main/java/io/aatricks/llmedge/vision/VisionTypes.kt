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

/**
 * Parameters for vision model analysis.
 *
 * @property maxTokens Maximum tokens to generate.
 * @property temperature Temperature for sampling.
 * @property systemPrompt Optional system prompt to prepend.
 */
data class VisionParams(
    val maxTokens: Int = 256,
    val temperature: Float = 0.2f,
    val systemPrompt: String? = null,
    /**
     * Number of embeddings per decode batch used when replaying prepared image embeddings.
     * If null, the adapter will choose a safe default (1).
     */
    val nBatch: Int? = null
)

/**
 * Result from vision model analysis.
 *
 * @property text The generated text from the vision model.
 * @property durationMs Processing time in milliseconds.
 * @property modelId The model identifier used.
 * @property tokensIn Number of input tokens.
 * @property tokensOut Number of output tokens.
 */
data class VisionResult(
    val text: String,
    val durationMs: Long,
    val modelId: String,
    val tokensIn: Int,
    val tokensOut: Int
)

data class VisionRuntimeMemory(
    val nativeBytes: Long,
    val stateBytes: Long,
)
