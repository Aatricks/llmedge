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
                instance.loadModel(
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
            ) = instance.setReasoningOptions(modelPtr, disableThinking, reasoningBudget)

            override fun addChatMessage(
                instance: SmolLM,
                modelPtr: Long,
                message: String,
                role: String,
            ) = instance.addChatMessage(modelPtr, message, role)

            override fun getResponseGenerationSpeed(
                instance: SmolLM,
                modelPtr: Long,
            ): Float = instance.getResponseGenerationSpeed(modelPtr)

            override fun getResponseGeneratedTokenCount(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.getResponseGeneratedTokenCount(modelPtr)

            override fun getResponseGenerationDurationMicros(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.getResponseGenerationDurationMicros(modelPtr)

            override fun getLastGenerationMetrics(
                instance: SmolLM,
                modelPtr: Long,
            ): SmolLM.GenerationMetrics {
                val packed = instance.nativeGetLastGenerationMetrics(modelPtr)
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
            ) = instance.nativeConfigureThreading(modelPtr, generationThreads, promptThreads)

            override fun getEstimatedNativeMemoryBytes(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.nativeGetEstimatedMemoryBytes(modelPtr)

            override fun getEstimatedStateMemoryBytes(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.nativeGetEstimatedStateMemoryBytes(modelPtr)

            override fun clearMessages(
                instance: SmolLM,
                modelPtr: Long,
            ) = instance.nativeClearMessages(modelPtr)

            override fun getContextSizeUsed(
                instance: SmolLM,
                modelPtr: Long,
            ): Int = instance.getContextSizeUsed(modelPtr)

            override fun getNativeModelPtr(
                instance: SmolLM,
                modelPtr: Long,
            ): Long = instance.getNativeModelPtr(modelPtr)

            override fun nativeDecodePreparedEmbeddings(
                instance: SmolLM,
                modelPtr: Long,
                embdPath: String,
                metaPath: String,
                nBatch: Int,
            ): Boolean = instance.nativeDecodePreparedEmbeddings(modelPtr, embdPath, metaPath, nBatch)

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
                instance.nativeDecodeEmbeddingsBuffer(
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
            ): Boolean = instance.nativePrimeImageBuffer(modelPtr, projectorNativePtr, imageData, nBatch)

            override fun getStateBytes(
                instance: SmolLM,
                modelPtr: Long,
            ): ByteArray? = instance.nativeGetStateBytes(modelPtr)

            override fun setStateBytes(
                instance: SmolLM,
                modelPtr: Long,
                state: ByteArray,
            ): Boolean = instance.nativeSetStateBytes(modelPtr, state)

            override fun getSequenceStateBytes(
                instance: SmolLM,
                modelPtr: Long,
                seqId: Int,
            ): ByteArray? = instance.nativeGetSequenceStateBytes(modelPtr, seqId)

            override fun setSequenceStateBytes(
                instance: SmolLM,
                modelPtr: Long,
                seqId: Int,
                state: ByteArray,
            ): Boolean = instance.nativeSetSequenceStateBytes(modelPtr, seqId, state)

            override fun startCompletion(
                instance: SmolLM,
                modelPtr: Long,
                prompt: String,
            ) = instance.startCompletion(modelPtr, prompt)

            override fun completionLoop(
                instance: SmolLM,
                modelPtr: Long,
            ): String = instance.completionLoop(modelPtr)

            override fun completionLoopBatch(
                instance: SmolLM,
                modelPtr: Long,
                maxTokens: Int,
            ): String = instance.completionLoopBatch(modelPtr, maxTokens)

            override fun completionLoopBatchBytes(
                instance: SmolLM,
                modelPtr: Long,
                maxTokens: Int,
            ): ByteArray? = instance.completionLoopBatchBytes(modelPtr, maxTokens)

            override fun stopCompletion(
                instance: SmolLM,
                modelPtr: Long,
            ) = instance.stopCompletion(modelPtr)

            override fun clearKvCache(
                instance: SmolLM,
                modelPtr: Long,
            ) = instance.nativeClearKvCache(modelPtr)

            override fun close(
                instance: SmolLM,
                modelPtr: Long,
            ) = instance.close(modelPtr)

            override fun hasVulkanBackendSupport(instance: SmolLM): Boolean =
                runCatching { instance.nativeHasVulkanBackendSupport() }
                    .getOrElse {
                        // Older desktop test binaries can miss newer JNI probe symbols.
                        // Fall back to process-level capability probing instead of failing load.
                        SmolLM.isVulkanBackendAvailable()
                    }
        }
    }
}
