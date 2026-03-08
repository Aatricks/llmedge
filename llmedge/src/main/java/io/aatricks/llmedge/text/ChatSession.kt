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
    private val history = mutableListOf<ConversationMessage>()

    suspend fun prepare() {
        client.prepare(model, options)
    }

    suspend fun reply(
        message: String,
        maxTokens: Int = -1,
        batchSize: Int = 1,
    ): String =
        sessionMutex.withLock {
            val runtime = client.acquire(model, options)
            history.add(ConversationMessage(ConversationRole.USER, message))
            val window = memory.trim(history)
            val prompt = PromptRenderer.render(window, systemPrompt)
            client
                .complete(
                    runtime = runtime,
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    options = options,
                    maxTokens = maxTokens,
                    batchSize = batchSize,
                ).also { reply ->
                    history.add(ConversationMessage(ConversationRole.ASSISTANT, reply))
                }
        }

    fun stream(message: String): Flow<TextStreamEvent> = flow {
        sessionMutex.withLock {
            val runtime = client.acquire(model, options)
            history.add(ConversationMessage(ConversationRole.USER, message))
            val window = memory.trim(history)
            val prompt = PromptRenderer.render(window, systemPrompt)
            val fullText = StringBuilder()
            emit(TextStreamEvent.Started(prompt))
            client.streamCompletion(runtime, prompt, systemPrompt, options).collect { chunk ->
                fullText.append(chunk)
                emit(TextStreamEvent.Chunk(chunk))
            }
            val response = fullText.toString()
            history.add(ConversationMessage(ConversationRole.ASSISTANT, response))
            emit(TextStreamEvent.Completed(response))
        }
    }

    fun historySnapshot(): List<ConversationMessage> = history.toList()
}
