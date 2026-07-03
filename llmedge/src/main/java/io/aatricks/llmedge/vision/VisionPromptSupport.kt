package io.aatricks.llmedge.vision

import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.GgufModelMetadata
import io.aatricks.llmedge.model.GgufModelMetadataSupport
import io.aatricks.llmedge.model.ModelSpec
import java.io.File

internal object VisionPromptSupport {
    private val visionMarkers = listOf("llava", "vision", "clip", "multimodal", "vlm", "mmproj")

    fun appearsVisionCapable(model: ModelSpec): Boolean =
        when (model) {
            is ModelSpec.LocalFile ->
                hasCapability(model, ModelCapability.VISION) ||
                    metadataIndicatesVision(model.file.absolutePath) == true ||
                    appearsVisionCapable(runCatching { model.file.absolutePath }.getOrElse { displayName(model) })
            is ModelSpec.HuggingFace ->
                hasCapability(model, ModelCapability.VISION) ||
                    appearsVisionCapable(displayName(model))
        }

    fun appearsVisionCapable(modelPath: String): Boolean {
        // A negative metadata result is NOT authoritative: a LLaVA base model's GGUF
        // architecture is just its LLM backbone (phi3/llama/mistral) — the vision
        // capability lives in the separate mmproj/projector. So the filename signal must
        // still be honored when metadata reads non-vision, rather than short-circuited.
        if (metadataIndicatesVision(modelPath) == true) {
            return true
        }
        return visionMarkers.any(File(modelPath).name.lowercase()::contains)
    }

    fun isProjectorSpec(projector: ModelSpec): Boolean =
        artifactKind(projector) == ModelArtifactKind.PROJECTOR ||
            hasCapability(projector, ModelCapability.PROJECTOR) ||
            projectorDisplayName(projector).contains("mmproj", ignoreCase = true) ||
            projectorDisplayName(projector).contains("projector", ignoreCase = true)

    fun hasProjectorSupport(projectorPath: String?): Boolean {
        if (projectorPath.isNullOrBlank()) {
            return false
        }
        val projectorFile = File(projectorPath)
        return projectorFile.exists() && projectorFile.isFile && projectorFile.canRead()
    }

    fun isReadyForMultimodalInference(
        model: ModelSpec,
        projector: ModelSpec?,
    ): Boolean = appearsVisionCapable(model) && projector != null && isProjectorSpec(projector)

    fun isReadyForMultimodalInference(modelPath: String, projectorPath: String?): Boolean =
        appearsVisionCapable(modelPath) && hasProjectorSupport(projectorPath)

    fun unsupportedReason(modelPath: String, projectorPath: String?): String =
        when {
            !appearsVisionCapable(modelPath) ->
                "Model '${File(modelPath).name}' does not appear to be a vision-capable GGUF. Provide a LLaVA/VLM-style model."
            !hasProjectorSupport(projectorPath) ->
                "Vision analysis requires a readable mmproj/projector file that matches the model. OCR remains available via VisionClient.extractText(...)."
            else ->
                "Vision model support is not ready for multimodal inference."
        }

    fun unsupportedReason(
        model: ModelSpec,
        projector: ModelSpec?,
    ): String =
        when {
            !appearsVisionCapable(model) ->
                "Model '${displayName(model)}' is not marked as vision-capable and does not match known VLM naming patterns."
            projector == null || !isProjectorSpec(projector) ->
                "Vision analysis requires a projector/mmproj model hint. OCR remains available via VisionClient.extractText(...)."
            else ->
                "Vision model support is not ready for multimodal inference."
        }

    fun formatVisionPrompt(prompt: String, imageFile: File): String {
        val normalized = prompt.trimStart()
        if (
            normalized.startsWith("SYSTEM:") ||
                normalized.startsWith("<|system|>") ||
                normalized.contains("OCR_TEXT_START") ||
                normalized.contains("EXAMPLES:") ||
                normalized.startsWith("<|user|>")
        ) {
            return prompt
        }

        return """
            [Image: ${imageFile.absolutePath}]

            User: $prompt
            Assistant:
        """.trimIndent()
    }

    fun estimateTokens(text: String): Int = (text.length / 4).coerceAtLeast(1)

    private fun metadataIndicatesVision(modelPath: String): Boolean? {
        val metadata = GgufModelMetadataSupport.inspect(modelPath) ?: return null
        return looksVisionCapable(metadata)
    }

    private fun looksVisionCapable(metadata: GgufModelMetadata): Boolean {
        val candidates =
            listOfNotNull(
                metadata.architecture,
                metadata.modelName,
            ).map(String::lowercase)
        return candidates.any { candidate -> visionMarkers.any(candidate::contains) }
    }

    private fun hasCapability(
        model: ModelSpec,
        capability: ModelCapability,
    ): Boolean = runCatching { model.hints.hasCapability(capability) }.getOrDefault(false)

    private fun artifactKind(model: ModelSpec): ModelArtifactKind? =
        runCatching { model.hints.artifactKind }.getOrNull()

    private fun displayName(model: ModelSpec): String =
        runCatching {
            when (model) {
                is ModelSpec.LocalFile -> model.file.name
                is ModelSpec.HuggingFace -> model.filename ?: model.repoId
            }
        }.getOrElse {
            runCatching { model.cacheKey }.getOrDefault("unknown-model")
        }

    private fun projectorDisplayName(projector: ModelSpec): String = displayName(projector)
}
