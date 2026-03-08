package io.aatricks.llmedge.text

sealed interface TextStreamEvent {
    data class Started(val prompt: String) : TextStreamEvent
    data class Chunk(val value: String) : TextStreamEvent
    data class Completed(val fullText: String) : TextStreamEvent
}
