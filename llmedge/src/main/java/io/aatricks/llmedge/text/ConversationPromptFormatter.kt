package io.aatricks.llmedge.text

internal object ConversationPromptFormatter {
    fun render(
        prefix: String,
        messages: List<ConversationMessage>,
        systemPrompt: String? = null,
    ): String =
        buildString {
            append(prefix)
            append("\n\n")
            systemPrompt
                ?.takeUnless(String::isBlank)
                ?.let {
                    append("System: ")
                    append(it.trim())
                    append('\n')
                }
            messages.forEach { message ->
                append(message.role.label)
                append(": ")
                append(message.content.trim())
                append('\n')
            }
        }.trimEnd()
}
