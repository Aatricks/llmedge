/*
 * Copyright (C) 2024 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.aatricks.llmedge

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Supported roles for [ChatSession] history replay. */
enum class Role {
    SYSTEM,
    USER,
    ASSISTANT,
}

/** A single chat turn stored by [ChatSession]. */
data class ChatMessage(
    val role: Role,
    val content: String,
)

/** Configuration for Kotlin-managed chat history replay. */
data class ChatSessionConfig(
    val maxHistoryMessages: Int = 6,
    val stripReasoningFromHistory: Boolean = true,
    val systemPrompt: String? = null,
) {
    init {
        require(maxHistoryMessages > 0) {
            "maxHistoryMessages must be greater than 0."
        }
    }
}

private val THINK_BLOCK_REGEX =
    Regex(
        pattern = "<think>.*?</think>",
        options = setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
    )

private val DANGLING_THINK_BLOCK_REGEX =
    Regex(
        pattern = "<think>.*$",
        options = setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
    )

/** Removes `<think>...</think>` blocks so replayed history consumes less context. */
fun String.stripThinkBlocks(): String {
    val withoutClosedBlocks = THINK_BLOCK_REGEX.replace(this, "")
    return DANGLING_THINK_BLOCK_REGEX.replace(withoutClosedBlocks, "").trim()
}

/**
 * Stateful chat helper that keeps the full transcript in Kotlin memory instead of relying on the
 * native KV cache for multi-turn history.
 *
 * Example:
 * ```kotlin
 * val smol = SmolLM()
 * smol.load(modelPath, SmolLM.InferenceParams(storeChats = false))
 *
 * val session = ChatSession(
 *     smol,
 *     ChatSessionConfig(
 *         maxHistoryMessages = 6,
 *         stripReasoningFromHistory = true,
 *         systemPrompt = "You are a concise assistant."
 *     )
 * )
 *
 * runBlocking {
 *     val reply = session.sendMessage("Explain KV cache in one paragraph.")
 *     session.sendMessageStream("Now summarize that in 3 bullets.").collect { chunk ->
 *         print(chunk)
 *     }
 * }
 *
 * smol.close()
 * ```
 */
class ChatSession(
    private val smol: SmolLM,
    private val config: ChatSessionConfig = ChatSessionConfig(),
) {
    private val sessionMutex = Mutex()
    private val history = mutableListOf<ChatMessage>()
    private val replaySystemPrompt = config.systemPrompt?.takeUnless(String::isBlank)
    private var storeChatsWarningLogged = false

    suspend fun sendMessage(message: String): String =
        sessionMutex.withLock {
            warnIfStoreChatsEnabled()
            history.add(ChatMessage(Role.USER, message))
            val activeWindow = buildActiveWindow()
            withContext(Dispatchers.IO) {
                rebuildNativeContext(activeWindow)
                smol.getResponse("")
            }.also { rawReply ->
                history.add(ChatMessage(Role.ASSISTANT, rawReply))
            }
        }

    fun sendMessageStream(message: String): Flow<String> = flow {
        sessionMutex.withLock {
            warnIfStoreChatsEnabled()
            history.add(ChatMessage(Role.USER, message))
            val activeWindow = buildActiveWindow()
            withContext(Dispatchers.IO) {
                rebuildNativeContext(activeWindow)
            }

            val rawReply = StringBuilder()
            var completed = false
            try {
                smol.getResponseAsFlow("").buffer(0).collect { chunk ->
                    rawReply.append(chunk)
                    emit(chunk)
                }
                completed = true
            } finally {
                if (completed) {
                    history.add(ChatMessage(Role.ASSISTANT, rawReply.toString()))
                }
            }
        }
    }

    internal val hasStoreChatsConflictWarning: Boolean
        get() = storeChatsWarningLogged

    internal fun historySnapshot(): List<ChatMessage> = history.toList()

    private fun buildActiveWindow(): List<ChatMessage> =
        history.takeLast(config.maxHistoryMessages).map { message ->
            if (config.stripReasoningFromHistory && message.role == Role.ASSISTANT) {
                message.copy(content = message.content.stripThinkBlocks())
            } else {
                message
            }
        }

    private fun rebuildNativeContext(activeWindow: List<ChatMessage>) {
        smol.clearKvCache()
        replaySystemPrompt?.let(smol::addSystemPrompt)
        activeWindow.forEach { message ->
            when (message.role) {
                Role.SYSTEM -> smol.addSystemPrompt(message.content)
                Role.USER -> smol.addUserMessage(message.content)
                Role.ASSISTANT -> smol.addAssistantMessage(message.content)
            }
        }
    }

    private fun warnIfStoreChatsEnabled() {
        if (storeChatsWarningLogged || smol.loadedInferenceParams?.storeChats != true) {
            return
        }
        storeChatsWarningLogged = true
        logSevere(
            "ChatSession expects SmolLM to be loaded with storeChats = false. " +
                "storeChats = true can conflict with Kotlin-managed history replay.",
        )
    }

    private fun logSevere(message: String) {
        try {
            Log.e(LOG_TAG, message)
        } catch (_: Throwable) {
            System.err.println("E/$LOG_TAG: $message")
        }
    }

    private companion object {
        const val LOG_TAG = "ChatSession"
    }
}
