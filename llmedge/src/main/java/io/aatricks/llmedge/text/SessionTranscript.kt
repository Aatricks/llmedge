package io.aatricks.llmedge.text

internal class SessionTranscript(
    private val memory: ConversationWindow,
) {
    private val history = mutableListOf<ConversationMessage>()

    fun previewWithUser(message: String): List<ConversationMessage> =
        memory.trim(history + ConversationMessage(ConversationRole.USER, message))

    fun commitTurn(
        message: String,
        response: String?,
    ) {
        history += ConversationMessage(ConversationRole.USER, message)
        response?.let { history += ConversationMessage(ConversationRole.ASSISTANT, it) }
    }

    fun clear() {
        history.clear()
    }

    fun snapshot(): List<ConversationMessage> = history.toList()
}
