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
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.runtime.SmolLM
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Compatibility adapter for callers that still hold a preloaded [SmolLM] instance.
 *
 * The authoritative multimodal path is [VisionPipeline]. This adapter now reuses the same input
 * preparation and generation helpers instead of keeping a second end-to-end implementation.
 */
class SmolLMVisionAdapter(
    private val context: Context,
    private val smolLM: SmolLM,
) : VisionModelAnalyzer {
    companion object {
        private const val TAG = "SmolLMVision"
        private const val JPEG_QUALITY = 90
    }

    private val inputPreparer = VisionInputPreparer(context, JPEG_QUALITY)
    private val runtimeExecutor = VisionRuntimeExecutor()

    private var modelPath: String? = null
    private var mmprojPath: String? = null
    private var hasVisionSupport = false
    private var projector: Projector? = null

    suspend fun loadVisionModel(
        modelPath: String,
        mmprojPath: String? = null,
        params: SmolLM.InferenceParams = SmolLM.InferenceParams(),
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
            closeProjector()
            this@SmolLMVisionAdapter.modelPath = modelPath
            this@SmolLMVisionAdapter.mmprojPath = mmprojPath

            smolLM.load(modelPath, params)

            val projectorPathValue =
                requireNotNull(mmprojPath) {
                    "Vision analysis requires a projector/mmproj file."
                }
            projector =
                Projector().also { loadedProjector ->
                    loadedProjector.init(projectorPathValue, smolLM.getNativeModelPointer())
                    check(loadedProjector.isReady()) {
                        "Native projector initialization failed for ${File(projectorPathValue).name}."
                    }
                }
            hasVisionSupport = true

            AndroidLogAdapter.d(TAG, "Loaded vision model: $modelPath")
            AndroidLogAdapter.d(TAG, "With mmproj: $projectorPathValue")
        } catch (e: Exception) {
            hasVisionSupport = false
            closeProjector()
            AndroidLogAdapter.e(TAG, "Failed to load vision model", e)
            throw e
        }
    }

    override suspend fun analyze(
        image: ImageSource,
        prompt: String,
        params: VisionParams,
    ): VisionResult =
        withContext(Dispatchers.IO) {
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
                val finalPrompt =
                    params.systemPrompt
                        ?.takeUnless(String::isBlank)
                        ?.let { systemPrompt -> "$systemPrompt\n\n$prompt" }
                        ?: prompt
                val preparedTurn = prepareTurn(image, finalPrompt)
                try {
                    val pipelineResult =
                        preparedTurn.request?.let { request ->
                            runtimeExecutor.execute(
                                request = request,
                                runtime = loadedRuntime(),
                                preparedInput = preparedTurn.preparedInput,
                                onStatus = null,
                                logStage = { _, _, _ -> },
                                maxTokens = params.maxTokens,
                            )
                        } ?: runtimeExecutor.execute(
                            prompt = finalPrompt,
                            runtime = loadedRuntime(allowPlaceholderProjector = true),
                            preparedInput = preparedTurn.preparedInput,
                            onStatus = null,
                            logStage = { _, _, _ -> },
                            maxTokens = params.maxTokens,
                        )
                    val duration = System.currentTimeMillis() - startTime
                    VisionResult(
                        text = pipelineResult.text,
                        durationMs = duration,
                        modelId = getModelId(),
                        tokensIn = estimateTokens(finalPrompt),
                        tokensOut = estimateTokens(pipelineResult.text),
                    )
                } finally {
                    preparedTurn.preparedInput.close()
                }
            } catch (e: Exception) {
                AndroidLogAdapter.e(TAG, "Vision analysis failed", e)
                when (e) {
                    is IllegalStateException,
                    is UnsupportedOperationException -> throw e
                    else -> throw IllegalStateException("Vision analysis failed: ${e.message}", e)
                }
            }
        }

    fun decodeEmbeddingsBuffer(
        embeddings: VisionEmbeddings,
        nBatch: Int = 1,
    ): Boolean {
        if (!hasVisionCapabilities()) return false
        return try {
            smolLM.decodeEmbeddingsBuffer(embeddings, nBatch)
        } catch (e: UnsatisfiedLinkError) {
            AndroidLogAdapter.w(TAG, "decodeEmbeddingsBuffer not available: ${e.message}")
            false
        }
    }

    override fun hasVisionCapabilities(): Boolean = hasVisionSupport

    override fun getModelId(): String = modelPath?.let { File(it).nameWithoutExtension } ?: "unknown"

    private fun checkVisionSupport(modelPath: String): Boolean =
        VisionPromptSupport.appearsVisionCapable(modelPath)

    private fun checkVisionSupport(
        modelPath: String,
        mmprojPath: String?,
    ): Boolean = VisionPromptSupport.isReadyForMultimodalInference(modelPath, mmprojPath)

    private fun formatVisionPrompt(
        prompt: String,
        imageFile: File,
    ): String = VisionPromptSupport.formatVisionPrompt(prompt, imageFile)

    private fun estimateTokens(text: String): Int = VisionPromptSupport.estimateTokens(text)

    private suspend fun prepareTurn(
        image: ImageSource,
        prompt: String,
    ): PreparedVisionTurn {
        val preparedEmbeddings = image.asPreparedEmbeddings()
        if (preparedEmbeddings != null) {
            return PreparedVisionTurn(
                request = null,
                preparedInput = preparedEmbeddings,
            )
        }

        val bitmap = ImageUtils.imageToBitmap(context, image)
        val request = buildRequest(prompt, bitmap)
        val preparedInput =
            inputPreparer.prepare(
                request = request,
                runtime = loadedRuntime(),
                onStatus = null,
                logStage = { _, _, _ -> },
            )
        return PreparedVisionTurn(request = request, preparedInput = preparedInput)
    }

    private fun ImageSource.asPreparedEmbeddings(): VisionPreparedInput.EmbeddingsFile? {
        val embedFile = (this as? ImageSource.FileSource)?.file ?: return null
        if (!embedFile.extension.equals("bin", ignoreCase = true)) {
            return null
        }
        val metaFile = File(embedFile.absolutePath + ".meta.json")
        check(metaFile.exists()) {
            "Prepared multimodal embeddings are missing. Ensure the projector mmproj file matches the model and native projector support is available."
        }
        AndroidLogAdapter.d(TAG, "Reusing prepared embeddings from ${embedFile.absolutePath}")
        return VisionPreparedInput.EmbeddingsFile(
            embedFile = embedFile,
            metaFile = metaFile,
            imageFile = embedFile,
            cleanupOnClose = false,
        )
    }

    private fun buildRequest(prompt: String, bitmap: android.graphics.Bitmap): VisionRequest =
        VisionRequest(
            image = bitmap,
            prompt = prompt,
            model = loadedModelSpec(),
            projector = loadedProjectorSpec(),
        )

    private fun loadedRuntime(allowPlaceholderProjector: Boolean = false): ManagedVisionRuntime {
        val loadedProjector =
            projector
                ?: if (allowPlaceholderProjector) {
                    Projector()
                } else {
                    throw IllegalStateException("Vision runtime is not loaded. Call loadVisionModel(...) first.")
                }
        val modelFile = requireNotNull(modelPath)
        val projectorFile = requireNotNull(mmprojPath)
        return ManagedVisionRuntime(
            fileSizeBytes = File(modelFile).length() + File(projectorFile).length(),
            smol = smolLM,
            projector = loadedProjector,
        )
    }

    private fun loadedModelSpec(): ModelSpec =
        ModelSpec.localFile(
            File(requireNotNull(modelPath)),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.GGUF_MODEL,
                    capabilities = setOf(ModelCapability.TEXT, ModelCapability.VISION),
                ),
        )

    private fun loadedProjectorSpec(): ModelSpec =
        ModelSpec.localFile(
            File(requireNotNull(mmprojPath)),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.PROJECTOR,
                    capabilities = setOf(ModelCapability.PROJECTOR),
                ),
        )

    fun close() {
        hasVisionSupport = false
        modelPath = null
        mmprojPath = null
        closeProjector()
    }

    private fun closeProjector() {
        projector?.close()
        projector = null
    }

    private data class PreparedVisionTurn(
        val request: VisionRequest?,
        val preparedInput: VisionPreparedInput,
    )
}

fun SmolLM.toVisionAdapter(context: Context): SmolLMVisionAdapter = SmolLMVisionAdapter(context, this)
