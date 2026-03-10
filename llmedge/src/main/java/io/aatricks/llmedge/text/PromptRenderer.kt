package io.aatricks.llmedge.text

object PromptRenderer {
    fun render(
        messages: List<ConversationMessage>,
        systemPrompt: String? = null,
    ): String =
        buildString {
            append("Continue the following conversation and reply as the assistant to the final user message.\n\n")
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

private val ConversationRole.label: String
    get() =
        when (this) {
            ConversationRole.SYSTEM -> "System"
            ConversationRole.USER -> "User"
            ConversationRole.ASSISTANT -> "Assistant"
        }
