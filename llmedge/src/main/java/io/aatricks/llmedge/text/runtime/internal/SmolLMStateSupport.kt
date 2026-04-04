package io.aatricks.llmedge.text.runtime.internal

import io.aatricks.llmedge.text.runtime.SmolLM

internal object SmolLMStateSupport {
    fun getStateBytes(instance: SmolLM): ByteArray? {
        val nativePtr = instance.requireLoadedHandle()
        return try {
            instance.bridge.getStateBytes(instance, nativePtr)
        } catch (_: UnsatisfiedLinkError) {
            null
        }
    }

    fun setStateBytes(instance: SmolLM, state: ByteArray): Boolean {
        val nativePtr = instance.requireLoadedHandle()
        return try {
            instance.bridge.setStateBytes(instance, nativePtr, state)
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    fun getSequenceStateBytes(instance: SmolLM, seqId: Int): ByteArray? {
        val nativePtr = instance.requireLoadedHandle()
        return try {
            instance.bridge.getSequenceStateBytes(instance, nativePtr, seqId)
        } catch (_: UnsatisfiedLinkError) {
            null
        }
    }

    fun setSequenceStateBytes(instance: SmolLM, seqId: Int, state: ByteArray): Boolean {
        val nativePtr = instance.requireLoadedHandle()
        return try {
            instance.bridge.setSequenceStateBytes(instance, nativePtr, seqId, state)
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    fun clearKvCache(instance: SmolLM) {
        val nativePtr = instance.requireLoadedHandle()
        try {
            instance.bridge.clearKvCache(instance, nativePtr)
        } catch (_: UnsatisfiedLinkError) {
        }
    }

    fun clearMessages(instance: SmolLM) {
        val nativePtr = instance.requireLoadedHandle()
        try {
            instance.bridge.clearMessages(instance, nativePtr)
        } catch (_: UnsatisfiedLinkError) {
        }
    }
}
