package io.aatricks.llmedge.tools

import io.aatricks.llmedge.text.stripThinkBlocks

internal class StreamingToolResponseCollector(
    private val onVisibleText: suspend (String) -> Unit,
) {
    private val rawResponse = StringBuilder()
    private var visibleResponse = ""
    private var finalTextDetected = false

    suspend fun append(chunk: String) {
        rawResponse.append(chunk)
        if (!finalTextDetected && shouldEmitVisibleText()) {
            finalTextDetected = true
        }
        if (finalTextDetected) {
            emitVisibleDelta()
        }
    }

    fun finish(): String = rawResponse.toString()

    private suspend fun emitVisibleDelta() {
        val currentVisible = rawResponse.toString().stripThinkBlocks()
        if (currentVisible == visibleResponse) {
            return
        }

        val delta =
            if (currentVisible.startsWith(visibleResponse)) {
                currentVisible.removePrefix(visibleResponse)
            } else {
                currentVisible
            }
        visibleResponse = currentVisible
        if (delta.isNotEmpty()) {
            onVisibleText(delta)
        }
    }

    private fun shouldEmitVisibleText(): Boolean {
        val trimmed = rawResponse.toString().stripThinkBlocks().trimStart()
        if (trimmed.isEmpty()) {
            return false
        }
        return !trimmed.startsWith("{") && !trimmed.startsWith("```")
    }
}
