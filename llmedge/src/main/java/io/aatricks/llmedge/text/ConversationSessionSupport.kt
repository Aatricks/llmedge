package io.aatricks.llmedge.text

import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

internal class ConversationSessionSupport(
    private val client: TextClient,
    private val model: ModelSpec,
    private val options: TextModelOptions,
    memory: ConversationWindow,
) {
    private val sessionMutex = Mutex()
    private val transcript = SessionTranscript(memory)

    suspend fun prepare() {
        client.prepare(model, options)
    }

    suspend fun <T> withRuntime(block: suspend ConversationRuntimeContext.() -> T): T =
        sessionMutex.withLock {
            val runtime = client.acquire(model, options)
            ConversationRuntimeContext(client, runtime, transcript, options).block()
        }

    fun historySnapshot(): List<ConversationMessage> = transcript.snapshot()

    fun withHistoryPreview(message: String): List<ConversationMessage> = transcript.previewWithUser(message)

    fun commitTurn(
        message: String,
        response: String?,
    ) {
        transcript.commitTurn(message, response)
    }
}

internal class ConversationRuntimeContext(
    private val client: TextClient,
    private val runtime: ManagedTextModel,
    private val transcript: SessionTranscript,
    private val options: TextModelOptions,
) {
    fun previewWithUser(message: String): List<ConversationMessage> = transcript.previewWithUser(message)

    fun commitTurn(
        message: String,
        response: String?,
    ) {
        transcript.commitTurn(message, response)
    }

    suspend fun complete(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        batchSize: Int,
    ): String = client.complete(runtime, prompt, systemPrompt, options, maxTokens, batchSize)

    fun streamCompletion(
        prompt: String,
        systemPrompt: String?,
        batchSize: Int,
    ) = client.streamCompletion(runtime, prompt, systemPrompt, options, batchSize)
}
