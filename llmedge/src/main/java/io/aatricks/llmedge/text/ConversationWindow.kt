package io.aatricks.llmedge.text

import kotlin.math.ceil

data class ConversationWindow(
    val maxTurns: Int = 6,
    val maxTokens: Int = 4096,
    val stripThinkTags: Boolean = true,
) {
    init {
        require(maxTurns > 0) { "maxTurns must be greater than 0." }
        require(maxTokens > 0) { "maxTokens must be greater than 0." }
    }

    fun trim(history: List<ConversationMessage>): List<ConversationMessage> {
        var window = history.takeLast(maxTurns * 2)
        while (window.isNotEmpty() && estimateTokenCount(window) > maxTokens) {
            window = window.drop(1)
        }
        return if (stripThinkTags) {
            window.map { message ->
                if (message.role == ConversationRole.ASSISTANT) {
                    message.copy(content = message.content.stripThinkBlocks())
                } else {
                    message
                }
            }
        } else {
            window
        }
    }

    fun estimateTokenCount(messages: List<ConversationMessage>): Int =
        messages.sumOf { estimateTokenCount(it.content) + 4 }

    fun estimateTokenCount(text: String): Int =
        ceil(text.trim().length.coerceAtLeast(1) / 4.0).toInt().coerceAtLeast(1)
}
