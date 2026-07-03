package io.aatricks.llmedge.text

internal class SessionTranscript(
    private val memory: ConversationWindow,
) {
    private val history = mutableListOf<ConversationMessage>()

    fun previewWithUser(message: String): List<ConversationMessage> = synchronized(this) {
        memory.trim(history + ConversationMessage(ConversationRole.USER, message))
    }

    fun commitTurn(
        message: String,
        response: String?,
    ) = synchronized(this) {
        history += ConversationMessage(ConversationRole.USER, message)
        response?.let { history += ConversationMessage(ConversationRole.ASSISTANT, it) }
    }

    fun clear() = synchronized(this) {
        history.clear()
    }

    fun snapshot(): List<ConversationMessage> = synchronized(this) {
        history.toList()
    }
}
