/*
 * Copyright (C) 2025 Shubham Panchal
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

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

class GGUFReader : Closeable {
    interface NativeBridge {
        fun getGGUFContextNativeHandle(modelPath: String): Long
        fun getContextSize(nativeHandle: Long): Long
        fun getChatTemplate(nativeHandle: Long): String
        fun getArchitecture(nativeHandle: Long): String
        fun getParameterCount(nativeHandle: Long): String
        fun getModelName(nativeHandle: Long): String
        fun releaseGGUFContext(nativeHandle: Long)
    }

    companion object {
        private var testBridgeProvider: ((GGUFReader) -> NativeBridge)? = null

        init {
            val disableNativeLoad = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")
            if (disableNativeLoad) {
                // Running under unit tests — skip native library load
                println("[GGUFReader] Native library load disabled via llmedge.disableNativeLoad")
            } else {
                try {
                    System.loadLibrary("ggufreader")
                } catch (e: UnsatisfiedLinkError) {
                    println("[GGUFReader] Native library missing or failed to load: ${e.message}")
                } catch (t: Throwable) {
                    println("[GGUFReader] Native library failed to load: ${'$'}{t.message}")
                }
            }
        }

        internal fun overrideNativeBridgeForTests(provider: (GGUFReader) -> NativeBridge) {
            testBridgeProvider = provider
        }

        internal fun resetNativeBridgeForTests() {
            testBridgeProvider = null
        }
    }

    private var nativeHandle: Long = 0L
    private val nativeBridge: NativeBridge = testBridgeProvider?.invoke(this) ?: DefaultNativeBridge()

    suspend fun load(modelPath: String) =
        withContext(Dispatchers.IO) {
            if (nativeHandle != 0L) {
                close()
            }
            nativeHandle = nativeBridge.getGGUFContextNativeHandle(modelPath)
        }

    fun getContextSize(): Long? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val contextSize = nativeBridge.getContextSize(nativeHandle)
        return if (contextSize == -1L) {
            null
        } else {
            contextSize
        }
    }

    fun getChatTemplate(): String? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val chatTemplate = nativeBridge.getChatTemplate(nativeHandle)
        return chatTemplate.ifEmpty {
            null
        }
    }

    /**
     * Read the model architecture from GGUF metadata (e.g., "wan", "llama", "stable-diffusion")
     */
    fun getArchitecture(): String? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val arch = nativeBridge.getArchitecture(nativeHandle)
        return arch.ifEmpty { null }
    }

    /**
     * Read parameter count metadata (e.g., "1.3B", "5B", "14B")
     * Falls back to estimating from model file size if not in metadata
     */
    fun getParameterCount(): String? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val params = nativeBridge.getParameterCount(nativeHandle)
        return params.ifEmpty { null }
    }

    /**
     * Read model name from GGUF metadata
     */
    fun getModelName(): String? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val name = nativeBridge.getModelName(nativeHandle)
        return name.ifEmpty { null }
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeBridge.releaseGGUFContext(nativeHandle)
            nativeHandle = 0L
        }
    }

    private class DefaultNativeBridge : NativeBridge {
        override external fun getGGUFContextNativeHandle(modelPath: String): Long
        override external fun getContextSize(nativeHandle: Long): Long
        override external fun getChatTemplate(nativeHandle: Long): String
        override external fun getArchitecture(nativeHandle: Long): String
        override external fun getParameterCount(nativeHandle: Long): String
        override external fun getModelName(nativeHandle: Long): String
        override external fun releaseGGUFContext(nativeHandle: Long)
    }
}
