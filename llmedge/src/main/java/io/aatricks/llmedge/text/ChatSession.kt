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

    suspend fun prepare() {
        support.prepare()
    }

    suspend fun reply(
        message: String,
        maxTokens: Int = -1,
        batchSize: Int = 0,
    ): String =
        support.withRuntime {
            // Every turn renders the full window: the pooled runtime runs with
            // storeChats=false, so no native chat state survives between turns.
            val window = previewWithUser(message)
            val prompt = PromptRenderer.render(window)
            val reply = complete(prompt, systemPrompt, maxTokens, batchSize)
            commitTurn(message, reply)
            reply
        }

    fun stream(message: String, batchSize: Int = 0): Flow<TextStreamEvent> = flow {
        support.withRuntime {
            val window = previewWithUser(message)
            val prompt = PromptRenderer.render(window)
            val fullText = StringBuilder()
            emit(TextStreamEvent.Started(prompt))
            streamCompletion(prompt, systemPrompt, batchSize).collect { chunk ->
                fullText.append(chunk)
                emit(TextStreamEvent.Chunk(chunk))
            }
            val response = fullText.toString()
            commitTurn(message, response)
            emit(TextStreamEvent.Completed(response))
        }
    }

    fun historySnapshot(): List<ConversationMessage> = support.historySnapshot()
}
