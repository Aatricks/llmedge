package io.aatricks.llmedge.vision

import java.io.File

internal object VisionPromptSupport {
    fun appearsVisionCapable(modelPath: String): Boolean {
        val modelName = File(modelPath).name.lowercase()
        return modelName.contains("llava") ||
            modelName.contains("vision") ||
            modelName.contains("clip") ||
            modelName.contains("multimodal")
    }

    fun formatVisionPrompt(prompt: String, imageFile: File): String {
        val normalized = prompt.trimStart()
        if (
            normalized.startsWith("SYSTEM:") ||
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