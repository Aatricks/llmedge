package io.aatricks.llmedge.text

enum class ConversationRole {
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL,
}

data class ConversationMessage(
    val role: ConversationRole,
    val content: String,
)

internal val ConversationRole.label: String
    get() =
        when (this) {
            ConversationRole.SYSTEM -> "System"
            ConversationRole.USER -> "User"
            ConversationRole.ASSISTANT -> "Assistant"
            ConversationRole.TOOL -> "Tool"
        }
