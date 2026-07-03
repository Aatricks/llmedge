package io.aatricks.llmedge.text.runtime

internal interface SmolLMNativeBridgeContract {
    fun loadModel(
        instance: SmolLM,
        modelPath: String,
        minP: Float,
        temperature: Float,
        storeChats: Boolean,
        contextSize: Long,
        chatTemplate: String,
        nThreads: Int,
        useMmap: Boolean,
        useMlock: Boolean,
        useVulkan: Boolean,
        useFlashAttn: Boolean,
        kvCacheTypeK: Int = -1,
        kvCacheTypeV: Int = -1,
        nGpuLayers: Int = 99,
    ): Long

    /**
     * Overload carrying the prompt micro-batch size (n_ubatch). The default
     * implementation drops it so bridges that predate the knob keep working.
     */
    fun loadModel(
        instance: SmolLM,
        modelPath: String,
        minP: Float,
        temperature: Float,
        storeChats: Boolean,
        contextSize: Long,
        chatTemplate: String,
        nThreads: Int,
        useMmap: Boolean,
        useMlock: Boolean,
        useVulkan: Boolean,
        useFlashAttn: Boolean,
        kvCacheTypeK: Int,
        kvCacheTypeV: Int,
        nGpuLayers: Int,
        nUbatch: Int,
    ): Long =
        loadModel(
            instance,
            modelPath,
            minP,
            temperature,
            storeChats,
            contextSize,
            chatTemplate,
            nThreads,
            useMmap,
            useMlock,
            useVulkan,
            useFlashAttn,
            kvCacheTypeK,
            kvCacheTypeV,
            nGpuLayers,
        )

    fun setReasoningOptions(
        instance: SmolLM,
        modelPtr: Long,
        disableThinking: Boolean,
        reasoningBudget: Int,
    )

    fun addChatMessage(
        instance: SmolLM,
        modelPtr: Long,
        message: String,
        role: String,
    )

    fun getResponseGenerationSpeed(
        instance: SmolLM,
        modelPtr: Long,
    ): Float

    fun getResponseGeneratedTokenCount(
        instance: SmolLM,
        modelPtr: Long,
    ): Long

    fun getResponseGenerationDurationMicros(
        instance: SmolLM,
        modelPtr: Long,
    ): Long

    fun getLastGenerationMetrics(
        instance: SmolLM,
        modelPtr: Long,
    ): SmolLM.GenerationMetrics {
        val elapsedMicros = getResponseGenerationDurationMicros(instance, modelPtr)
        val tokenCount = getResponseGeneratedTokenCount(instance, modelPtr)
        val tokensPerSecond =
            if (elapsedMicros <= 0L || tokenCount <= 0L) {
                0f
            } else {
                getResponseGenerationSpeed(instance, modelPtr)
            }
        return SmolLM.GenerationMetrics(
            tokensPerSecond = tokensPerSecond,
            tokenCount = tokenCount,
            elapsedMicros = elapsedMicros,
        )
    }

    fun configureThreading(
        instance: SmolLM,
        modelPtr: Long,
        generationThreads: Int,
        promptThreads: Int,
    ) = Unit

    fun getEstimatedNativeMemoryBytes(
        instance: SmolLM,
        modelPtr: Long,
    ): Long = 0L

    fun getEstimatedStateMemoryBytes(
        instance: SmolLM,
        modelPtr: Long,
    ): Long = 0L

    fun clearMessages(
        instance: SmolLM,
        modelPtr: Long,
    ) = Unit

    fun getContextSizeUsed(
        instance: SmolLM,
        modelPtr: Long,
    ): Int

    fun wasContextLimitReached(
        instance: SmolLM,
        modelPtr: Long,
    ): Boolean = false

    fun getNativeModelPtr(
        instance: SmolLM,
        modelPtr: Long,
    ): Long

    fun nativeDecodePreparedEmbeddings(
        instance: SmolLM,
        modelPtr: Long,
        embdPath: String,
        metaPath: String,
        nBatch: Int,
    ): Boolean

    fun nativeDecodeEmbeddingsBuffer(
        instance: SmolLM,
        modelPtr: Long,
        embeddings: FloatArray,
        nTokens: Int,
        nx: Int,
        ny: Int,
        embdDim: Int,
        useMrope: Boolean,
        useNonCausal: Boolean,
        nBatch: Int,
    ): Boolean = false

    fun nativePrimeImageBuffer(
        instance: SmolLM,
        modelPtr: Long,
        projectorNativePtr: Long,
        imageData: ByteArray,
        nBatch: Int,
    ): Boolean = false

    fun getStateBytes(
        instance: SmolLM,
        modelPtr: Long,
    ): ByteArray? = null

    fun setStateBytes(
        instance: SmolLM,
        modelPtr: Long,
        state: ByteArray,
    ): Boolean = false

    fun getSequenceStateBytes(
        instance: SmolLM,
        modelPtr: Long,
        seqId: Int,
    ): ByteArray? = null

    fun setSequenceStateBytes(
        instance: SmolLM,
        modelPtr: Long,
        seqId: Int,
        state: ByteArray,
    ): Boolean = false

    fun close(
        instance: SmolLM,
        modelPtr: Long,
    )

    fun startCompletion(
        instance: SmolLM,
        modelPtr: Long,
        prompt: String,
    )

    fun completionLoop(
        instance: SmolLM,
        modelPtr: Long,
    ): String

    fun completionLoopBatch(
        instance: SmolLM,
        modelPtr: Long,
        maxTokens: Int,
    ): String

    fun completionLoopBatchBytes(
        instance: SmolLM,
        modelPtr: Long,
        maxTokens: Int,
    ): ByteArray? = null

    fun stopCompletion(
        instance: SmolLM,
        modelPtr: Long,
    )

    fun clearKvCache(
        instance: SmolLM,
        modelPtr: Long,
    )

    fun hasVulkanBackendSupport(instance: SmolLM): Boolean = true
}
