package io.aatricks.llmedge.text.runtime.internal

import io.aatricks.llmedge.text.runtime.SmolLM

internal object SmolLMStateSupport {
    fun getStateBytes(instance: SmolLM): ByteArray? {
        val nativePtr = instance.supportRequireHandle()
        return try {
            instance.supportNativeBridge.getStateBytes(instance, nativePtr)
        } catch (_: UnsatisfiedLinkError) {
            null
        }
    }

    fun setStateBytes(instance: SmolLM, state: ByteArray): Boolean {
        val nativePtr = instance.supportRequireHandle()
        return try {
            instance.supportNativeBridge.setStateBytes(instance, nativePtr, state)
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    fun getSequenceStateBytes(instance: SmolLM, seqId: Int): ByteArray? {
        val nativePtr = instance.supportRequireHandle()
        return try {
            instance.supportNativeBridge.getSequenceStateBytes(instance, nativePtr, seqId)
        } catch (_: UnsatisfiedLinkError) {
            null
        }
    }

    fun setSequenceStateBytes(instance: SmolLM, seqId: Int, state: ByteArray): Boolean {
        val nativePtr = instance.supportRequireHandle()
        return try {
            instance.supportNativeBridge.setSequenceStateBytes(instance, nativePtr, seqId, state)
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    fun clearKvCache(instance: SmolLM) {
        val nativePtr = instance.supportRequireHandle()
        try {
            instance.supportNativeBridge.clearKvCache(instance, nativePtr)
        } catch (_: UnsatisfiedLinkError) {
        }
    }

    fun clearMessages(instance: SmolLM) {
        val nativePtr = instance.supportRequireHandle()
        try {
            instance.supportNativeBridge.clearMessages(instance, nativePtr)
        } catch (_: UnsatisfiedLinkError) {
        }
    }
}
