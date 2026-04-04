package io.aatricks.llmedge.text.runtime.internal

import io.aatricks.llmedge.text.runtime.SmolLM
import io.aatricks.llmedge.vision.VisionEmbeddings

internal object SmolLMVisionInterop {
    fun decodePreparedEmbeddings(
        instance: SmolLM,
        embdPath: String,
        metaPath: String,
        nBatch: Int,
    ): Boolean {
        val nativePtr = instance.requireLoadedHandle()
        return try {
            instance.bridge.nativeDecodePreparedEmbeddings(
                instance,
                nativePtr,
                embdPath,
                metaPath,
                nBatch,
            )
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    fun decodeEmbeddingsBuffer(
        instance: SmolLM,
        embeddings: VisionEmbeddings,
        nBatch: Int,
    ): Boolean {
        val nativePtr = instance.requireLoadedHandle()
        return try {
            instance.bridge.nativeDecodeEmbeddingsBuffer(
                instance,
                nativePtr,
                embeddings.data,
                embeddings.nTokens,
                embeddings.nx,
                embeddings.ny,
                embeddings.embdDim,
                embeddings.useMrope,
                embeddings.useNonCausal,
                nBatch,
            )
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    fun primeImageBuffer(
        instance: SmolLM,
        projectorNativePtr: Long,
        imageData: ByteArray,
        nBatch: Int,
    ): Boolean {
        val nativePtr = instance.requireLoadedHandle()
        return try {
            instance.bridge.nativePrimeImageBuffer(
                instance,
                nativePtr,
                projectorNativePtr,
                imageData,
                nBatch,
            )
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }
}
