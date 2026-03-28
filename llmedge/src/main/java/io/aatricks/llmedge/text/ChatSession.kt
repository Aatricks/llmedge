package io.aatricks.llmedge.text

import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class ChatSession internal constructor(
    private val client: TextClient,
    private val model: ModelSpec,
    private val memory: ConversationWindow,
    private val systemPrompt: String?,
    private val options: TextModelOptions,
) {
    private val sessionMutex = Mutex()
    private val transcript = SessionTranscript(memory)
    private var lastStateSnapshot: ByteArray? = null
    private var snapshotWindowSize: Int = 0

    suspend fun prepare() {
        client.prepare(model, options)
    }

    suspend fun reply(
        message: String,
        maxTokens: Int = -1,
        batchSize: Int = 0,
    ): String =
        sessionMutex.withLock {
            val runtime = client.acquire(model, options)
            val window = transcript.previewWithUser(message)

            // Incremental path: restore from snapshot when the window grew by exactly
            // one message (the new user message) and no old messages were trimmed.
            val canRestore = lastStateSnapshot != null && window.size == snapshotWindowSize + 1

            val (reply, newState) = if (canRestore) {
                client.chatTurn(
                    runtime = runtime,
                    prompt = message,
                    systemPrompt = null,
                    options = options,
                    maxTokens = maxTokens,
                    batchSize = batchSize,
                    restoreState = lastStateSnapshot,
                )
            } else {
                val prompt = PromptRenderer.render(window)
                client.chatTurn(
                    runtime = runtime,
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    options = options,
                    maxTokens = maxTokens,
                    batchSize = batchSize,
                    restoreState = null,
                )
            }

            transcript.commitTurn(message, reply)
            lastStateSnapshot = newState
            snapshotWindowSize = window.size + 1
            reply
        }

    fun stream(message: String, batchSize: Int = 0): Flow<TextStreamEvent> = flow {
        sessionMutex.withLock {
            val runtime = client.acquire(model, options)
            val window = transcript.previewWithUser(message)
            val prompt = PromptRenderer.render(window)
            val fullText = StringBuilder()
            emit(TextStreamEvent.Started(prompt))
            // Streaming path: always does full replay (state snapshot not feasible mid-stream)
            client.streamCompletion(runtime, prompt, systemPrompt, options, batchSize).collect { chunk ->
                fullText.append(chunk)
                emit(TextStreamEvent.Chunk(chunk))
            }
            val response = fullText.toString()
            transcript.commitTurn(message, response)
            // Invalidate snapshot after streaming since we can't capture state mid-stream
            lastStateSnapshot = null
            snapshotWindowSize = 0
            emit(TextStreamEvent.Completed(response))
        }
    }

    fun historySnapshot(): List<ConversationMessage> = transcript.snapshot()
}
