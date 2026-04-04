package io.aatricks.llmedge.text

object PromptRenderer {
    private const val DEFAULT_PREFIX =
        "Continue the following conversation and reply as the assistant to the final user message."

    fun render(
        messages: List<ConversationMessage>,
        systemPrompt: String? = null,
    ): String = ConversationPromptFormatter.render(DEFAULT_PREFIX, messages, systemPrompt)
}
