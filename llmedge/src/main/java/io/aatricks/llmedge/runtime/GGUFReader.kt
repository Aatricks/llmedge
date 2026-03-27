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

package io.aatricks.llmedge.runtime

import io.aatricks.llmedge.core.InvalidModelStateException
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.core.NativeBridgeProvider
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeLibraryLoader
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.model.ModelFileValidator
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
        fun getFileType(nativeHandle: Long): Int = -1
        fun getDominantTensorType(nativeHandle: Long): Int = -1
        fun releaseGGUFContext(nativeHandle: Long)
    }

    companion object {
        private const val LOG_TAG = "GGUFReader"
        private val nativeBridgeProvider = NativeBridgeProvider<GGUFReader, NativeBridge> { DefaultNativeBridge() }

        init {
            NativeLibraryLoader.ensureGgufReaderLoaded(
                required = false,
                onDebug = { message -> AndroidLogAdapter.d(LOG_TAG, message) },
                onError = { message, throwable -> AndroidLogAdapter.e(LOG_TAG, message, throwable) },
            )
        }

        internal fun overrideNativeBridgeForTests(provider: (GGUFReader) -> NativeBridge) {
            nativeBridgeProvider.override(provider)
        }

        internal fun resetNativeBridgeForTests() {
            nativeBridgeProvider.reset()
        }
    }

    private var nativeHandle: Long = 0L
    private val nativeBridge: NativeBridge = nativeBridgeProvider.create(this)

    suspend fun load(modelPath: String) =
        withContext(Dispatchers.IO) {
            if (nativeHandle != 0L) {
                close()
            }
            val validatedModel = ModelFileValidator.requireGgufFile(modelPath, "GGUF model")
            nativeHandle =
                NativeCall.requireHandle(
                    NativeCall.binding(
                        "ggufreader",
                        "GGUF metadata reader JNI bindings are unavailable.",
                    ) {
                        nativeBridge.getGGUFContextNativeHandle(validatedModel.absolutePath)
                    },
                    validatedModel.absolutePath,
                    "GGUF metadata reader returned an invalid native handle.",
                )
        }

    fun getContextSize(): Long? {
        verifyHandle()
        val contextSize = nativeBridge.getContextSize(nativeHandle)
        return if (contextSize == -1L) {
            null
        } else {
            contextSize
        }
    }

    fun getChatTemplate(): String? {
        verifyHandle()
        val chatTemplate = nativeBridge.getChatTemplate(nativeHandle)
        return chatTemplate.ifEmpty {
            null
        }
    }

    /**
     * Read the model architecture from GGUF metadata (e.g., "wan", "llama", "stable-diffusion")
     */
    fun getArchitecture(): String? {
        verifyHandle()
        val arch = nativeBridge.getArchitecture(nativeHandle)
        return arch.ifEmpty { null }
    }

    /**
     * Read parameter count metadata (e.g., "1.3B", "5B", "14B")
     * Falls back to estimating from model file size if not in metadata
     */
    fun getParameterCount(): String? {
        verifyHandle()
        val params = nativeBridge.getParameterCount(nativeHandle)
        return params.ifEmpty { null }
    }

    /**
     * Read model name from GGUF metadata
     */
    fun getModelName(): String? {
        verifyHandle()
        val name = nativeBridge.getModelName(nativeHandle)
        return name.ifEmpty { null }
    }

    internal fun getFileType(): Int? {
        verifyHandle()
        val fileType = nativeBridge.getFileType(nativeHandle)
        return if (fileType < 0) null else fileType
    }

    internal fun getDominantTensorType(): Int? {
        verifyHandle()
        val dominantTensorType = nativeBridge.getDominantTensorType(nativeHandle)
        return if (dominantTensorType < 0) null else dominantTensorType
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeBridge.releaseGGUFContext(nativeHandle)
            nativeHandle = 0L
        }
    }

    private fun verifyHandle() {
        if (nativeHandle == 0L) {
            throw InvalidModelStateException("Use GGUFReader.load() to initialize the reader")
        }
    }

    private class DefaultNativeBridge : NativeBridge {
        override external fun getGGUFContextNativeHandle(modelPath: String): Long
        override external fun getContextSize(nativeHandle: Long): Long
        override external fun getChatTemplate(nativeHandle: Long): String
        override external fun getArchitecture(nativeHandle: Long): String
        override external fun getParameterCount(nativeHandle: Long): String
        override external fun getModelName(nativeHandle: Long): String
        override external fun getFileType(nativeHandle: Long): Int
        override external fun getDominantTensorType(nativeHandle: Long): Int
        override external fun releaseGGUFContext(nativeHandle: Long)
    }
}
