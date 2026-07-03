package io.aatricks.llmedge.text.runtime

import io.aatricks.llmedge.text.runtime.internal.SmolLMCompletionSupport
import io.aatricks.llmedge.text.runtime.internal.SmolLMStateSupport
import io.aatricks.llmedge.text.runtime.internal.SmolLMVisionInterop
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow

internal object SmolLMFacadeOperations {
    fun addUserMessage(instance: SmolLM, message: String) {
        val nativePtr = instance.requireLoadedHandle()
        instance.bridge.addChatMessage(instance, nativePtr, message, "user")
    }

    fun addSystemPrompt(instance: SmolLM, prompt: String) {
        val nativePtr = instance.requireLoadedHandle()
        instance.bridge.addChatMessage(instance, nativePtr, prompt, "system")
    }

    fun addAssistantMessage(instance: SmolLM, message: String) {
        val nativePtr = instance.requireLoadedHandle()
        instance.bridge.addChatMessage(instance, nativePtr, message, "assistant")
    }

    fun getResponseGenerationSpeed(instance: SmolLM): Float {
        val nativePtr = instance.requireLoadedHandle()
        return instance.bridge.getResponseGenerationSpeed(instance, nativePtr)
    }

    fun getLastGenerationMetrics(instance: SmolLM): SmolLM.GenerationMetrics {
        val nativePtr = instance.requireLoadedHandle()
        return instance.bridge.getLastGenerationMetrics(instance, nativePtr)
    }

    fun getEstimatedNativeMemoryBytes(instance: SmolLM): Long {
        val nativePtr = instance.requireLoadedHandle()
        return instance.bridge.getEstimatedNativeMemoryBytes(instance, nativePtr)
    }

    fun getEstimatedStateMemoryBytes(instance: SmolLM): Long {
        val nativePtr = instance.requireLoadedHandle()
        return instance.bridge.getEstimatedStateMemoryBytes(instance, nativePtr)
    }

    fun getContextLengthUsed(instance: SmolLM): Int {
        val nativePtr = instance.requireLoadedHandle()
        return instance.bridge.getContextSizeUsed(instance, nativePtr)
    }

    fun wasContextLimitReached(instance: SmolLM): Boolean {
        val nativePtr = instance.requireLoadedHandle()
        return instance.bridge.wasContextLimitReached(instance, nativePtr)
    }

    fun getResponseAsFlow(instance: SmolLM, query: String): Flow<String> =
        getResponseAsFlow(instance, query, Dispatchers.IO)

    fun getResponseAsFlow(
        instance: SmolLM,
        query: String,
        dispatcher: CoroutineDispatcher,
    ): Flow<String> = getResponseAsFlow(instance, query, dispatcher, DEFAULT_STREAM_BATCH)

    // Emitting a few tokens per native call amortizes the JNI round-trip without
    // noticeably chunking the stream (4 tokens is well under typical frame budgets).
    private const val DEFAULT_STREAM_BATCH = 1

    fun getResponseAsFlow(
        instance: SmolLM,
        query: String,
        dispatcher: CoroutineDispatcher,
        batchSize: Int,
    ): Flow<String> = SmolLMCompletionSupport.getResponseAsFlow(instance, query, dispatcher, batchSize)

    fun getResponse(
        instance: SmolLM,
        query: String,
        maxTokens: Int,
        batchSize: Int,
    ): String = SmolLMCompletionSupport.getResponse(instance, query, maxTokens, batchSize)

    fun stopCompletion(instance: SmolLM) {
        SmolLMCompletionSupport.stopCompletion(instance)
    }

    fun getNativeModelPointer(instance: SmolLM): Long {
        val nativePtr = instance.requireLoadedHandle()
        return instance.bridge.getNativeModelPtr(instance, nativePtr)
    }

    fun decodePreparedEmbeddings(
        instance: SmolLM,
        embdPath: String,
        metaPath: String,
        nBatch: Int,
    ): Boolean = SmolLMVisionInterop.decodePreparedEmbeddings(instance, embdPath, metaPath, nBatch)

    fun decodeEmbeddingsBuffer(
        instance: SmolLM,
        embeddings: io.aatricks.llmedge.vision.VisionEmbeddings,
        nBatch: Int,
    ): Boolean = SmolLMVisionInterop.decodeEmbeddingsBuffer(instance, embeddings, nBatch)

    fun primeImageBuffer(
        instance: SmolLM,
        projectorNativePtr: Long,
        imageData: ByteArray,
        nBatch: Int,
    ): Boolean = SmolLMVisionInterop.primeImageBuffer(instance, projectorNativePtr, imageData, nBatch)

    fun getStateBytes(instance: SmolLM): ByteArray? = SmolLMStateSupport.getStateBytes(instance)

    fun setStateBytes(instance: SmolLM, state: ByteArray): Boolean =
        SmolLMStateSupport.setStateBytes(instance, state)

    fun getSequenceStateBytes(instance: SmolLM, seqId: Int): ByteArray? =
        SmolLMStateSupport.getSequenceStateBytes(instance, seqId)

    fun setSequenceStateBytes(
        instance: SmolLM,
        seqId: Int,
        state: ByteArray,
    ): Boolean = SmolLMStateSupport.setSequenceStateBytes(instance, seqId, state)

    fun clearKvCache(instance: SmolLM) {
        SmolLMStateSupport.clearKvCache(instance)
    }

    fun clearMessages(instance: SmolLM) {
        SmolLMStateSupport.clearMessages(instance)
    }
}
