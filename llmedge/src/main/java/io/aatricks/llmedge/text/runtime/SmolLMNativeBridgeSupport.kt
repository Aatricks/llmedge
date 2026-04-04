package io.aatricks.llmedge.text.runtime

internal object SmolLMNativeBridgeSupport {
    fun defaultProvider(): (SmolLM) -> SmolLM.NativeBridge = { instance ->
        object : SmolLM.NativeBridge {
            override fun loadModel(
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
            ): Long =
                instance.bridgeLoadModel(
                    modelPath,
                    minP,
                    temperature,
                    storeChats,
                    contextSize,
                    chatTemplate,
                    nThreads,
                    useMmap,
                    useMlock,
                    instance.resolveRequestedBackendForLoad(useVulkan).id,
                    useFlashAttn,
                    kvCacheTypeK,
                    kvCacheTypeV,
                    nGpuLayers,
                )

            override fun setReasoningOptions(
                instance: SmolLM,
                modelPtr: Long,
                disableThinking: Boolean,
                reasoningBudget: Int,
            ) = instance.bridgeSetReasoningOptions(modelPtr, disableThinking, reasoningBudget)

            override fun addChatMessage(
                instance: SmolLM,
                modelPtr: Long,
                message: String,
                role: String,
            ) = instance.bridgeAddChatMessage(modelPtr, message, role)

            override fun getResponseGenerationSpeed(
                instance: SmolLM,
                modelPtr: Long,
            ): Float = instance.bridgeGetResponseGenerationSpeed(modelPtr)

            override fun getResponseGeneratedTokenCount(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.bridgeGetResponseGeneratedTokenCount(modelPtr)

            override fun getResponseGenerationDurationMicros(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.bridgeGetResponseGenerationDurationMicros(modelPtr)

            override fun getLastGenerationMetrics(
                instance: SmolLM,
                modelPtr: Long,
            ): SmolLM.GenerationMetrics {
                val packed = instance.bridgeGetLastGenerationMetricsPacked(modelPtr)
                if (packed == null || packed.size < 3) {
                    return super<SmolLM.NativeBridge>.getLastGenerationMetrics(instance, modelPtr)
                }
                val elapsedMicros = packed[0]
                val tokenCount = packed[1]
                val tokensPerSecond =
                    if (elapsedMicros <= 0L || tokenCount <= 0L) {
                        0f
                    } else {
                        Float.fromBits(packed[2].toInt())
                    }
                return SmolLM.GenerationMetrics(
                    tokensPerSecond = tokensPerSecond,
                    tokenCount = tokenCount,
                    elapsedMicros = elapsedMicros,
                )
            }

            override fun configureThreading(
                instance: SmolLM,
                modelPtr: Long,
                generationThreads: Int,
                promptThreads: Int,
            ) = instance.bridgeConfigureThreading(modelPtr, generationThreads, promptThreads)

            override fun getEstimatedNativeMemoryBytes(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.bridgeGetEstimatedNativeMemoryBytes(modelPtr)

            override fun getEstimatedStateMemoryBytes(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.bridgeGetEstimatedStateMemoryBytes(modelPtr)

            override fun clearMessages(
                instance: SmolLM,
                modelPtr: Long,
            ) = instance.bridgeClearMessages(modelPtr)

            override fun getContextSizeUsed(
                instance: SmolLM,
                modelPtr: Long,
            ): Int = instance.bridgeGetContextSizeUsed(modelPtr)

            override fun getNativeModelPtr(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.bridgeGetNativeModelPtr(modelPtr)

            override fun nativeDecodePreparedEmbeddings(
                instance: SmolLM,
                modelPtr: Long,
                embdPath: String,
                metaPath: String,
                nBatch: Int,
            ): Boolean = instance.bridgeDecodePreparedEmbeddings(modelPtr, embdPath, metaPath, nBatch)

            override fun nativeDecodeEmbeddingsBuffer(
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
            ): Boolean =
                instance.bridgeDecodeEmbeddingsBuffer(
                    modelPtr,
                    embeddings,
                    nTokens,
                    nx,
                    ny,
                    embdDim,
                    useMrope,
                    useNonCausal,
                    nBatch,
                )

            override fun nativePrimeImageBuffer(
                instance: SmolLM,
                modelPtr: Long,
                projectorNativePtr: Long,
                imageData: ByteArray,
                nBatch: Int,
            ): Boolean = instance.bridgePrimeImageBuffer(modelPtr, projectorNativePtr, imageData, nBatch)

            override fun getStateBytes(
                instance: SmolLM,
                modelPtr: Long,
            ): ByteArray? = instance.bridgeGetStateBytes(modelPtr)

            override fun setStateBytes(
                instance: SmolLM,
                modelPtr: Long,
                state: ByteArray,
            ): Boolean = instance.bridgeSetStateBytes(modelPtr, state)

            override fun getSequenceStateBytes(
                instance: SmolLM,
                modelPtr: Long,
                seqId: Int,
            ): ByteArray? = instance.bridgeGetSequenceStateBytes(modelPtr, seqId)

            override fun setSequenceStateBytes(
                instance: SmolLM,
                modelPtr: Long,
                seqId: Int,
                state: ByteArray,
            ): Boolean = instance.bridgeSetSequenceStateBytes(modelPtr, seqId, state)

            override fun startCompletion(
                instance: SmolLM,
                modelPtr: Long,
                prompt: String,
            ) = instance.bridgeStartCompletion(modelPtr, prompt)

            override fun completionLoop(
                instance: SmolLM,
                modelPtr: Long,
            ): String = instance.bridgeCompletionLoop(modelPtr)

            override fun completionLoopBatch(
                instance: SmolLM,
                modelPtr: Long,
                maxTokens: Int,
            ): String = instance.bridgeCompletionLoopBatch(modelPtr, maxTokens)

            override fun completionLoopBatchBytes(
                instance: SmolLM,
                modelPtr: Long,
                maxTokens: Int,
            ): ByteArray? = instance.bridgeCompletionLoopBatchBytes(modelPtr, maxTokens)

            override fun stopCompletion(
                instance: SmolLM,
                modelPtr: Long,
            ) = instance.bridgeStopCompletion(modelPtr)

            override fun clearKvCache(
                instance: SmolLM,
                modelPtr: Long,
            ) = instance.bridgeClearKvCache(modelPtr)

            override fun close(
                instance: SmolLM,
                modelPtr: Long,
            ) = instance.bridgeClose(modelPtr)

            override fun hasVulkanBackendSupport(instance: SmolLM): Boolean =
                instance.bridgeHasVulkanBackendSupport()
        }
    }
}
