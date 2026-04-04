package io.aatricks.llmedge.text

import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow

class ChatSession internal constructor(
    private val client: TextClient,
    private val model: ModelSpec,
    private val memory: ConversationWindow,
    private val systemPrompt: String?,
    private val options: TextModelOptions,
) {
    private val support = ConversationSessionSupport(client, model, options, memory)
    private var lastStateSnapshot: ByteArray? = null
    private var snapshotWindowSize: Int = 0

    suspend fun prepare() {
        support.prepare()
    }

    suspend fun reply(
        message: String,
        maxTokens: Int = -1,
        batchSize: Int = 0,
    ): String =
        support.withRuntime {
            val window = previewWithUser(message)

            // Incremental path: restore from snapshot when the window grew by exactly
            // one message (the new user message) and no old messages were trimmed.
            val canRestore = lastStateSnapshot != null && window.size == snapshotWindowSize + 1

            val (reply, newState) = if (canRestore) {
                chatTurn(
                    prompt = message,
                    systemPrompt = null,
                    maxTokens = maxTokens,
                    batchSize = batchSize,
                    restoreState = lastStateSnapshot,
                    maxStateBytes = 64L * 1024L * 1024L,
                )
            } else {
                val prompt = PromptRenderer.render(window)
                chatTurn(
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    maxTokens = maxTokens,
                    batchSize = batchSize,
                    restoreState = null,
                    maxStateBytes = 64L * 1024L * 1024L,
                )
            }

            commitTurn(message, reply)
            lastStateSnapshot = newState
            snapshotWindowSize = window.size + 1
            reply
        }

    fun stream(message: String, batchSize: Int = 0): Flow<TextStreamEvent> = flow {
        support.withRuntime {
            val window = previewWithUser(message)
            val prompt = PromptRenderer.render(window)
            val fullText = StringBuilder()
            emit(TextStreamEvent.Started(prompt))
            // Streaming path: always does full replay (state snapshot not feasible mid-stream)
            streamCompletion(prompt, systemPrompt, batchSize).collect { chunk ->
                fullText.append(chunk)
                emit(TextStreamEvent.Chunk(chunk))
            }
            val response = fullText.toString()
            commitTurn(message, response)
            // Invalidate snapshot after streaming since we can't capture state mid-stream
            lastStateSnapshot = null
            snapshotWindowSize = 0
            emit(TextStreamEvent.Completed(response))
        }
    }

    fun historySnapshot(): List<ConversationMessage> = support.historySnapshot()
}
