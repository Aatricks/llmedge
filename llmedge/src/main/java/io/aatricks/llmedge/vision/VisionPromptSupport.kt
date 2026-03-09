package io.aatricks.llmedge.vision

import java.io.File

internal object VisionPromptSupport {
    private val visionMarkers = listOf("llava", "vision", "clip", "multimodal", "vlm", "mmproj")

    fun appearsVisionCapable(modelPath: String): Boolean {
        val modelName = File(modelPath).name.lowercase()
        return visionMarkers.any(modelName::contains)
    }

    fun hasProjectorSupport(projectorPath: String?): Boolean {
        if (projectorPath.isNullOrBlank()) {
            return false
        }
        val projectorFile = File(projectorPath)
        return projectorFile.exists() && projectorFile.isFile && projectorFile.canRead()
    }

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
}