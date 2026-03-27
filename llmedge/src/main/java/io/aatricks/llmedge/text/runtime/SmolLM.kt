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

package io.aatricks.llmedge.text.runtime

import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.runtime.BackendRuntimePolicy

import android.content.Context
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.InvalidModelStateException
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.core.NativeBridgeProvider
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeLibraryLoader
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.model.ModelFileValidator
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext

/**
 * Kotlin wrapper for the native LLM runtime. Handles loading models and providing a simple API for
 * running completions and managing model state.
 */
class SmolLM(useVulkan: Boolean = true) : AutoCloseable {

    internal interface NativeBridge {
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

        fun setReasoningOptions(
                instance: SmolLM,
                modelPtr: Long,
                disableThinking: Boolean,
                reasoningBudget: Int
        )
        fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String)
        fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float
        fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long
        fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long
        fun getLastGenerationMetrics(instance: SmolLM, modelPtr: Long): GenerationMetrics {
            val elapsedMicros = getResponseGenerationDurationMicros(instance, modelPtr)
            val tokenCount = getResponseGeneratedTokenCount(instance, modelPtr)
            val tokensPerSecond =
                if (elapsedMicros <= 0L || tokenCount <= 0L) {
                    0f
                } else {
                    getResponseGenerationSpeed(instance, modelPtr)
                }
            return GenerationMetrics(
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
        fun getEstimatedNativeMemoryBytes(instance: SmolLM, modelPtr: Long): Long = 0L
        fun getEstimatedStateMemoryBytes(instance: SmolLM, modelPtr: Long): Long = 0L
        fun clearMessages(instance: SmolLM, modelPtr: Long) = Unit
        fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int
        fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long
        fun nativeDecodePreparedEmbeddings(
                instance: SmolLM,
                modelPtr: Long,
                embdPath: String,
                metaPath: String,
                nBatch: Int
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
                nBatch: Int
        ): Boolean = false
        fun nativePrimeImageBuffer(
                instance: SmolLM,
                modelPtr: Long,
                projectorNativePtr: Long,
                imageData: ByteArray,
                nBatch: Int
        ): Boolean = false
        fun getStateBytes(instance: SmolLM, modelPtr: Long): ByteArray? = null
        fun setStateBytes(instance: SmolLM, modelPtr: Long, state: ByteArray): Boolean = false
        fun getSequenceStateBytes(instance: SmolLM, modelPtr: Long, seqId: Int): ByteArray? = null
        fun setSequenceStateBytes(instance: SmolLM, modelPtr: Long, seqId: Int, state: ByteArray): Boolean = false
        fun close(instance: SmolLM, modelPtr: Long)
        fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String)
        fun completionLoop(instance: SmolLM, modelPtr: Long): String
        fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String
        /** Batched completion returning raw UTF-8 bytes to avoid per-call NewStringUTF overhead. */
        fun completionLoopBatchBytes(instance: SmolLM, modelPtr: Long, maxTokens: Int): ByteArray? = null
        fun stopCompletion(instance: SmolLM, modelPtr: Long)
        fun clearKvCache(instance: SmolLM, modelPtr: Long)
        fun hasVulkanBackendSupport(instance: SmolLM): Boolean = true
    }
    companion object {
        private const val LOG_TAG = "SmolLM"
        private const val DEFAULT_CONTEXT_SIZE_CAP: Long = 8_192L
        private const val MIN_CONTEXT_SIZE: Long = 1_024L
        private const val DEFAULT_REASONING_BUDGET: Int = -1
        private val GGUF_FILE_TYPE_NAMES =
            mapOf(
                138 to "IQ2_K",
                139 to "IQ3_K",
                140 to "IQ4_K",
                141 to "IQ5_K",
                142 to "IQ6_K",
                149 to "Q8_KV",
            )
        private val GGUF_TENSOR_TYPE_NAMES =
            mapOf(
                137 to "IQ2_K",
                138 to "IQ3_K",
                139 to "IQ4_K",
                140 to "IQ5_K",
                141 to "IQ6_K",
                151 to "Q8_KV",
            )
        /** Device-aware batch size: scales with P-core count for optimal JNI throughput. */
        val DEFAULT_BLOCKING_BATCH_SIZE: Int =
            CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.TOKEN_GENERATION)
                .coerceIn(4, 16)

        private fun logD(tag: String, message: String) = AndroidLogAdapter.d(tag, message)

        private fun logI(tag: String, message: String) = AndroidLogAdapter.i(tag, message)

        private fun logW(tag: String, message: String) = AndroidLogAdapter.w(tag, message)

        private fun logE(tag: String, message: String, throwable: Throwable? = null) =
            AndroidLogAdapter.e(tag, message, throwable)

        @JvmStatic
        fun isOpenClAvailable(): Boolean =
            try {
                nativeIsOpenClAvailable()
            } catch (_: Throwable) {
                false
            }

        @JvmStatic
        fun isVulkanBackendAvailable(): Boolean =
            try {
                nativeIsVulkanAvailable()
            } catch (_: Throwable) {
                true
            }

        init {
            NativeLibraryLoader.ensureSmolLMLoaded(
                required = true,
                onDebug = { message -> logD(LOG_TAG, message) },
                onError = { message, throwable -> logE(LOG_TAG, message, throwable) },
            )
        }

        private val defaultNativeBridgeProvider: (SmolLM) -> NativeBridge = { instance ->
            object : NativeBridge {
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
                                nGpuLayers
                        )

                override fun setReasoningOptions(
                        instance: SmolLM,
                        modelPtr: Long,
                        disableThinking: Boolean,
                        reasoningBudget: Int
                ) = instance.setReasoningOptions(modelPtr, disableThinking, reasoningBudget)

                override fun addChatMessage(
                        instance: SmolLM,
                        modelPtr: Long,
                        message: String,
                        role: String
                ) = instance.addChatMessage(modelPtr, message, role)

                override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float =
                        instance.getResponseGenerationSpeed(modelPtr)
                override fun getResponseGeneratedTokenCount(
                        instance: SmolLM,
                        modelPtr: Long
                ): Long = instance.getResponseGeneratedTokenCount(modelPtr)
                override fun getResponseGenerationDurationMicros(
                        instance: SmolLM,
                        modelPtr: Long
                ): Long = instance.getResponseGenerationDurationMicros(modelPtr)
                override fun getLastGenerationMetrics(
                    instance: SmolLM,
                    modelPtr: Long,
                ): GenerationMetrics {
                    val packed = instance.nativeGetLastGenerationMetrics(modelPtr)
                    if (packed == null || packed.size < 3) {
                        return super<NativeBridge>.getLastGenerationMetrics(instance, modelPtr)
                    }
                    val elapsedMicros = packed[0]
                    val tokenCount = packed[1]
                    val tokensPerSecondBits = packed[2].toInt()
                    val tokensPerSecond =
                        if (elapsedMicros <= 0L || tokenCount <= 0L) {
                            0f
                        } else {
                            Float.fromBits(tokensPerSecondBits)
                        }
                    return GenerationMetrics(
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
                override fun getEstimatedNativeMemoryBytes(instance: SmolLM, modelPtr: Long): Long =
                    instance.nativeGetEstimatedMemoryBytes(modelPtr)
                override fun getEstimatedStateMemoryBytes(instance: SmolLM, modelPtr: Long): Long =
                    instance.nativeGetEstimatedStateMemoryBytes(modelPtr)
                override fun clearMessages(instance: SmolLM, modelPtr: Long) =
                    instance.nativeClearMessages(modelPtr)
                override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int =
                        instance.getContextSizeUsed(modelPtr)
                override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long =
                        instance.getNativeModelPtr(modelPtr)
                override fun nativeDecodePreparedEmbeddings(
                        instance: SmolLM,
                        modelPtr: Long,
                        embdPath: String,
                        metaPath: String,
                        nBatch: Int
                ): Boolean =
                        instance.nativeDecodePreparedEmbeddings(
                                modelPtr,
                                embdPath,
                                metaPath,
                                nBatch
                        )
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
                        nBatch: Int
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
                                nBatch
                        )
                override fun close(instance: SmolLM, modelPtr: Long) = instance.close(modelPtr)
                override fun nativePrimeImageBuffer(
                        instance: SmolLM,
                        modelPtr: Long,
                        projectorNativePtr: Long,
                        imageData: ByteArray,
                        nBatch: Int
                ): Boolean =
                        instance.nativePrimeImageBuffer(modelPtr, projectorNativePtr, imageData, nBatch)
                override fun getStateBytes(instance: SmolLM, modelPtr: Long): ByteArray? =
                        instance.nativeGetStateBytes(modelPtr)
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
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) =
                        instance.startCompletion(modelPtr, prompt)
                override fun completionLoop(instance: SmolLM, modelPtr: Long): String =
                        instance.completionLoop(modelPtr)
                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String =
                        instance.completionLoopBatch(modelPtr, maxTokens)
                override fun completionLoopBatchBytes(instance: SmolLM, modelPtr: Long, maxTokens: Int): ByteArray? =
                        instance.completionLoopBatchBytes(modelPtr, maxTokens)
                override fun stopCompletion(instance: SmolLM, modelPtr: Long) =
                        instance.stopCompletion(modelPtr)
                override fun clearKvCache(instance: SmolLM, modelPtr: Long) =
                    instance.nativeClearKvCache(modelPtr)
                override fun hasVulkanBackendSupport(instance: SmolLM): Boolean =
                    instance.nativeHasVulkanBackendSupport()
            }
        }

        private val nativeBridgeProvider = NativeBridgeProvider(defaultNativeBridgeProvider)

        internal fun overrideNativeBridgeForTests(provider: (SmolLM) -> NativeBridge) {
            nativeBridgeProvider.override(provider)
        }

        internal fun resetNativeBridgeForTests() {
            nativeBridgeProvider.reset()
        }

        @JvmStatic
        private external fun nativeIsOpenClAvailable(): Boolean

        @JvmStatic
        private external fun nativeIsVulkanAvailable(): Boolean

        internal fun createLoadedForTests(
                nativePtr: Long,
                useVulkan: Boolean = false,
                loadedParams: InferenceParams = InferenceParams(),
        ): SmolLM {
            val s = SmolLM(useVulkan)
            s.nativePtr = nativePtr
            s.loadedInferenceParams = loadedParams
            s.selectedBackend = if (useVulkan) ComputeBackend.VULKAN else ComputeBackend.CPU
            return s
        }
    }

    private var nativePtr = 0L
    private val nativeBridge: NativeBridge = Companion.nativeBridgeProvider.create(this)
    private var useVulkanGPU = true
    private var requestedLoadBackend: ComputeBackend? = null
    private var selectedBackend: ComputeBackend = ComputeBackend.CPU
    private var currentThinkingMode = ThinkingMode.DEFAULT
    private var currentReasoningBudget = DEFAULT_REASONING_BUDGET
    internal var loadedInferenceParams: InferenceParams? = null
        private set

    init {
        this.useVulkanGPU = useVulkan
        this.selectedBackend = if (useVulkan) ComputeBackend.VULKAN else ComputeBackend.CPU
    }

    /** Returns true if this SmolLM instance will try to use Vulkan-backed GPU layers. */
    fun isVulkanEnabled(): Boolean =
        if (nativePtr != 0L) {
            selectedBackend == ComputeBackend.VULKAN
        } else {
            useVulkanGPU
        }

    internal fun getActiveBackend(): ComputeBackend =
        if (nativePtr != 0L) {
            selectedBackend
        } else {
            requestedLoadBackend ?: if (useVulkanGPU) ComputeBackend.VULKAN else ComputeBackend.CPU
        }

    internal fun setPreferredBackendForLoad(backend: ComputeBackend?) {
        requestedLoadBackend = backend
    }

    internal fun resolveRequestedBackendForLoad(legacyUseVulkan: Boolean): ComputeBackend =
        requestedLoadBackend ?: if (legacyUseVulkan) ComputeBackend.VULKAN else ComputeBackend.CPU

    /**
     * Provides default values for inference parameters. These values are used when the
     * corresponding parameters are not provided by the user or are not available in the GGUF model
     * file.
     */
    object DefaultInferenceParams {
        val contextSize: Long = 1024L
        val chatTemplate: String =
                "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|> ' }}{% endif %}{{'<|im_start|>' + message['role'] + ' ' + message['content'] + '<|im_end|>' + ' '}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant ' }}{% endif %}"
    }

    enum class ThinkingMode {
        DEFAULT,
        DISABLED;

        internal val disableReasoning: Boolean
            get() = this == DISABLED

        internal val reasoningBudget: Int
            get() = if (this == DISABLED) 0 else DEFAULT_REASONING_BUDGET
    }

    enum class KvCacheType(internal val nativeCode: Int) {
        DEFAULT(0),
        F16(1),
        Q8_0(2),
        Q4_0(3),
        Q8_KV(4),
    }

    /**
     * Data class to hold the inference parameters for the LLM.
     *
     * @property minP The minimum probability for a token to be considered.
     * ```
     *                Also known as top-P sampling. (Default: 0.1f)
     * @property temperature
     * ```
     * The temperature for sampling. Higher values make the output more random.
     * ```
     *                       (Default: 0.8f)
     * @property storeChats
     * ```
     * Whether to store the chat history in memory. If true, the LLM will
     * ```
     *                      remember previous interactions in the current session. (Default: true)
     * @property contextSize
     * ```
     * The context size (in tokens) for the LLM. This determines how much
     * ```
     *                       of the previous conversation the LLM can "remember". If null, the
     *                       value from the GGUF model file will be used, or a default value if
     *                       not present in the model file. (Default: null)
     * @property chatTemplate
     * ```
     * The chat template to use for formatting the conversation. This
     * ```
     *                        is a Jinja2 template string. If null, the value from the GGUF
     *                        model file will be used, or a default value if not present in the
     *                        model file. (Default: null)
     * @property numThreads
     * ```
    * The number of threads to use for prompt/batch processing. (Default: 4)
    * @property generationThreads
    * ```
    * Optional thread count for single-token generation. If omitted, [numThreads]
    * is reused for both prompt and generation phases.
     * @property useMmap Whether to use memory-mapped file I/O for loading the model.
     * ```
     *                   This can improve loading times and reduce memory usage. (Default: true)
     * @property useMlock
     * ```
     * Whether to lock the model in memory. This can prevent the model from
     * ```
     *                    being swapped out to disk, potentially improving performance. (Default: false)
     * @property thinkingMode
     * ```
     * Controls whether reasoning “think” traces remain enabled. Use
     * ```
     *                        [ThinkingMode.DISABLED] to request the equivalent of llama.cpp's
     *                        `--no-think` flag. (Default: [ThinkingMode.DEFAULT])
     * @property reasoningBudget
     * ```
     * Optional override for llama.cpp's `--reasoning-budget` flag. Set to
     * ```
     *                           `0` to disable thinking explicitly, `-1` to leave it unrestricted,
     *                           or omit to let [thinkingMode] decide.
     * ```
     */
        data class InferenceParams(
                val minP: Float = 0.1f,
                val temperature: Float = 0.8f,
                @Deprecated(
                        message = "Use Kotlin-managed ChatSession for multi-turn state instead of native storeChats.",
                        replaceWith = ReplaceWith("false"),
                )
                val storeChats: Boolean = true,
                val contextSize: Long? = null,
                val chatTemplate: String? = null,
            val numThreads: Int = 4,
            val generationThreads: Int? = null,
            val useMmap: Boolean = true,
            val useMlock: Boolean = false,
            val useFlashAttn: Boolean = true,
            val thinkingMode: ThinkingMode = ThinkingMode.DEFAULT,
            val reasoningBudget: Int? = null,
            /** Stable llmedge KV cache type for keys. Backend-specific ggml_type mapping happens natively. */
            val kvCacheTypeK: KvCacheType = KvCacheType.DEFAULT,
            /** Stable llmedge KV cache type for values. Backend-specific ggml_type mapping happens natively. */
            val kvCacheTypeV: KvCacheType = KvCacheType.DEFAULT,
            /** Number of layers to offload to GPU. Only used when Vulkan is enabled. Default 99 = all layers. */
            val nGpuLayers: Int = 99,
    )

    /**
     * Summary of the most recent response generation.
     *
     * @property tokensPerSecond Average decoding throughput for the response.
     * @property tokenCount Number of tokens emitted for the response.
     * @property elapsedMicros Total decoding time in microseconds.
     */
    data class GenerationMetrics(
            val tokensPerSecond: Float,
            val tokenCount: Long,
            val elapsedMicros: Long,
    ) {
        val elapsedMillis: Double
            get() = elapsedMicros / 1_000.0

        val elapsedSeconds: Double
            get() = elapsedMicros / 1_000_000.0
    }

    /**
     * Loads the GGUF model from the given path. This function will read the metadata from the GGUF
     * model file, such as the context size and chat template, and use them if they are not
     * explicitly provided in the `params`.
     *
     * @param modelPath The path to the GGUF model file.
     * @param params The inference parameters to use. If not provided, default values will be used.
     * ```
     *               If `contextSize` or `chatTemplate` are not provided in `params`,
     *               the values from the GGUF model file will be used. If those are also
     *               not available in the model file, then default values from [DefaultInferenceParams]
     *               will be used.
     * @return
     * ```
     * `true` if the model was loaded successfully, `false` otherwise.
     * @throws io.aatricks.llmedge.core.ModelFileNotFoundException if the model file does not exist.
     * @throws io.aatricks.llmedge.core.InvalidModelFileException if the file is unreadable, empty,
     * or not a GGUF model.
     * @throws io.aatricks.llmedge.core.ModelLoadException if the native runtime fails to load the
     * validated model.
     */
    suspend fun load(
            modelPath: String,
            params: InferenceParams = InferenceParams(),
    ) =
            withContext(Dispatchers.IO) {
                val validatedModel = ModelFileValidator.requireGgufFile(modelPath, "SmolLM model")
                if (nativePtr != 0L) {
                    close()
                }

                val ggufReader = GGUFReader()
                val resolvedContextSize: Long
                val resolvedChatTemplate: String
                val fileType: Int?
                val dominantTensorType: Int?
                try {
                    ggufReader.load(validatedModel.absolutePath)
                    val modelContextSize =
                            ggufReader.getContextSize() ?: DefaultInferenceParams.contextSize
                    resolvedContextSize = resolveContextSize(params.contextSize, modelContextSize)
                    resolvedChatTemplate = resolveChatTemplate(params.chatTemplate, ggufReader)
                    fileType = ggufReader.getFileType()
                    dominantTensorType = ggufReader.getDominantTensorType()
                } finally {
                    ggufReader.close()
                }
                @Suppress("DEPRECATION")
                val storeChats = params.storeChats
                val promptThreads = params.numThreads.coerceAtLeast(1)
                val backendCandidates =
                    requestedLoadBackend?.let(::listOf)
                        ?: BackendRuntimePolicy.candidates(
                            subsystem = ComputeSubsystem.TEXT,
                            allowGpu = useVulkanGPU,
                            openClAvailable = isOpenClAvailable(),
                            vulkanAvailable = isVulkanBackendAvailable(),
                        )

                var lastLoadError: Throwable? = null
                nativePtr = 0L
                for (backend in backendCandidates) {
                    requestedLoadBackend = backend
                    try {
                        val candidateHandle =
                            NativeCall.binding("smollm", "SmolLM JNI bindings are unavailable.") {
                                nativeBridge.loadModel(
                                    this@SmolLM,
                                    validatedModel.absolutePath,
                                    params.minP,
                                    params.temperature,
                                    storeChats,
                                    resolvedContextSize,
                                    resolvedChatTemplate,
                                    promptThreads,
                                    params.useMmap,
                                    params.useMlock,
                                    backend == ComputeBackend.VULKAN,
                                    params.useFlashAttn,
                                    params.kvCacheTypeK.nativeCode,
                                    params.kvCacheTypeV.nativeCode,
                                    params.nGpuLayers,
                                )
                            }
                        nativePtr =
                            NativeCall.requireHandle(
                                candidateHandle,
                                validatedModel.absolutePath,
                                "The native SmolLM loader returned an invalid handle.",
                            )
                        selectedBackend = backend
                        break
                    } catch (e: NativeBindingException) {
                        requestedLoadBackend = null
                        throw e
                    } catch (e: IllegalStateException) {
                        lastLoadError =
                            ModelLoadException(
                                validatedModel.absolutePath,
                                e.message ?: "The native SmolLM loader reported an unknown error.",
                                e,
                            )
                    } catch (e: ModelLoadException) {
                        lastLoadError = e
                    }

                    if (backend != ComputeBackend.CPU) {
                        BackendRuntimePolicy.blacklist(ComputeSubsystem.TEXT, backend)
                        logW(LOG_TAG, "Failed to load SmolLM on $backend; retrying with the next backend")
                    }
                }
                requestedLoadBackend = null
                if (nativePtr == 0L) {
                    throw (lastLoadError
                        ?: ModelLoadException(
                            validatedModel.absolutePath,
                            "The native SmolLM loader returned an invalid handle.",
                        ))
                }
                val generationThreads = (params.generationThreads ?: promptThreads).coerceAtLeast(1)
                nativeBridge.configureThreading(this@SmolLM, nativePtr, generationThreads, promptThreads)
                val reasoningBudget =
                        resolvedReasoningBudget(params.thinkingMode, params.reasoningBudget)
                applyReasoningState(params.thinkingMode, reasoningBudget)

                // Pin inference threads to performance cores on big.LITTLE SoCs
                val pCoreMask = CpuTopology.getPerformanceCoreMask()
                if (pCoreMask != 0L) {
                    setThreadAffinity(nativePtr, pCoreMask)
                }
                loadedInferenceParams =
                        params.copy(
                                contextSize = resolvedContextSize,
                                chatTemplate = resolvedChatTemplate,
                        numThreads = promptThreads,
                        generationThreads = generationThreads,
                        )
            }

    /**
     * Downloads a GGUF model from Hugging Face (if needed) and loads it for inference.
     *
     * @param context Android context used to resolve the destination directory under app storage.
     * @param modelId Hugging Face repository id (for example, "unsloth/Qwen3-0.6B-GGUF").
     * @param revision Repository revision or branch name. Defaults to "main".
     * @param preferredQuantizations Ordered list of substrings used to pick the desired GGUF
     * variant.
     * @param filename Optional explicit file name/path (relative to the repo root) to download.
     * @param params Inference parameters to apply once the model is loaded.
     * @param token Optional Hugging Face access token for private repositories.
     * @param forceDownload When true, always redownload the file even if a cached copy exists.
     * @param preferSystemDownloader When true, prefer Android's DownloadManager for large
     * downloads.
     * @param onProgress Optional progress listener receiving downloaded bytes and total bytes (when
     * known).
     *
     * @return [HuggingFaceHub.ModelDownloadResult] describing the loaded asset.
     */
    suspend fun loadFromHuggingFace(
            context: Context,
            modelId: String,
            revision: String = "main",
            preferredQuantizations: List<String> = HuggingFaceHub.DEFAULT_QUANTIZATION_PRIORITIES,
            filename: String? = null,
            params: InferenceParams = InferenceParams(),
            token: String? = null,
            forceDownload: Boolean = false,
            preferSystemDownloader: Boolean = true,
            onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): HuggingFaceHub.ModelDownloadResult {
        val downloadResult =
                HuggingFaceHub.ensureModelOnDisk(
                        context = context,
                        modelId = modelId,
                        revision = revision,
                        preferredQuantizations = preferredQuantizations,
                        filename = filename,
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = preferSystemDownloader,
                        onProgress = onProgress,
                )
        load(downloadResult.file.absolutePath, params)
        return downloadResult
    }

    /**
     * Adds a user message to the chat history. This message will be considered as part of the
     * conversation when generating the next response.
     *
     * @param message The user's message.
     * @throws IllegalStateException if the model is not loaded.
     */
    fun addUserMessage(message: String) {
        verifyHandle()
        nativeBridge.addChatMessage(this, nativePtr, message, "user")
    }

    /** Adds the system prompt for the LLM */
    fun addSystemPrompt(prompt: String) {
        verifyHandle()
        nativeBridge.addChatMessage(this, nativePtr, prompt, "system")
    }

    /**
     * Adds the assistant message for LLM inference An assistant message is the response given by
     * the LLM for a previous query in the conversation
     */
    fun addAssistantMessage(message: String) {
        verifyHandle()
        nativeBridge.addChatMessage(this, nativePtr, message, "assistant")
    }

    fun getThinkingMode(): ThinkingMode = currentThinkingMode

    fun getReasoningBudget(): Int = currentReasoningBudget

    fun isThinkingEnabled(): Boolean = currentReasoningBudget != 0

    fun setThinkingMode(mode: ThinkingMode) {
        verifyHandle()
        val targetBudget = if (mode.disableReasoning) 0 else DEFAULT_REASONING_BUDGET
        applyReasoningState(mode, targetBudget)
    }

    fun setThinkingEnabled(enabled: Boolean) {
        setThinkingMode(if (enabled) ThinkingMode.DEFAULT else ThinkingMode.DISABLED)
    }

    fun setReasoningBudget(budget: Int) {
        verifyHandle()
        val mode = if (budget == 0) ThinkingMode.DISABLED else ThinkingMode.DEFAULT
        applyReasoningState(mode, budget)
    }

    /**
     * Returns the rate (in tokens per second) at which the LLM generated its last response via
     * `getResponse()`
     */
    fun getResponseGenerationSpeed(): Float {
        verifyHandle()
        return nativeBridge.getResponseGenerationSpeed(this, nativePtr)
    }

    /**
     * Returns throughput information for the last completed response. The metrics are reset on the
     * next call to [getResponse] or [getResponseAsFlow].
     */
    fun getLastGenerationMetrics(): GenerationMetrics {
        verifyHandle()
        return nativeBridge.getLastGenerationMetrics(this, nativePtr)
    }

    fun getEstimatedNativeMemoryBytes(): Long {
        verifyHandle()
        return nativeBridge.getEstimatedNativeMemoryBytes(this, nativePtr)
    }

    fun getEstimatedStateMemoryBytes(): Long {
        verifyHandle()
        return nativeBridge.getEstimatedStateMemoryBytes(this, nativePtr)
    }

    /**
     * Returns the number of tokens consumed by the LLM's context window The context of the LLM is
     * roughly the output of, tokenize(apply_chat_template(messages_in_conversation))
     */
    fun getContextLengthUsed(): Int {
        verifyHandle()
        return nativeBridge.getContextSizeUsed(this, nativePtr)
    }

    /**
     * Return the LLM response to the given query as an async Flow. This is useful for streaming the
     * response as it is generated by the LLM.
     *
     * @param query The query to ask the LLM.
     * @return A Flow of Strings, where each String is a piece of the response.
     * ```
     *         The flow completes when the LLM has finished generating the response.
     *         The special token "[EOG]" (End Of Generation) indicates the end of the response.
     * @throws IllegalStateException
     * ```
     * if the model is not loaded.
     */
    fun getResponseAsFlow(query: String): Flow<String> = getResponseAsFlow(query, Dispatchers.IO)

    fun getResponseAsFlow(query: String, dispatcher: CoroutineDispatcher): Flow<String> =
            getResponseAsFlow(query, dispatcher, 1)

    fun getResponseAsFlow(
        query: String,
        dispatcher: CoroutineDispatcher,
        batchSize: Int,
    ): Flow<String> =
            flow {
                        verifyHandle()
                        try {
                            nativeBridge.startCompletion(this@SmolLM, nativePtr, query)
                            if (batchSize > 1) {
                                // Use raw byte-based JNI path to avoid per-batch NewStringUTF
                                // overhead. The byte path returns null from the default interface
                                // method when not implemented by test bridges, so we try it first
                                // and fall back to the String path.
                                val eogBytes = "[EOG]".toByteArray(Charsets.UTF_8)
                                var bytes = try {
                                    nativeBridge.completionLoopBatchBytes(this@SmolLM, nativePtr, batchSize)
                                } catch (_: Throwable) { null }

                                if (bytes != null) {
                                    // Byte-based fast path
                                    while (!bytes.contentEquals(eogBytes) && bytes!!.isNotEmpty()) {
                                        currentCoroutineContext().ensureActive()
                                        emit(String(bytes!!, Charsets.UTF_8))
                                        bytes = nativeBridge.completionLoopBatchBytes(this@SmolLM, nativePtr, batchSize)
                                        if (bytes == null) break
                                    }
                                } else {
                                    // Fallback: String-based path (test bridges)
                                    var piece = nativeBridge.completionLoopBatch(this@SmolLM, nativePtr, batchSize)
                                    while (piece != "[EOG]" && piece.isNotEmpty()) {
                                        currentCoroutineContext().ensureActive()
                                        emit(piece)
                                        piece = nativeBridge.completionLoopBatch(this@SmolLM, nativePtr, batchSize)
                                    }
                                }
                            } else {
                                var piece = nativeBridge.completionLoop(this@SmolLM, nativePtr)
                                while (piece != "[EOG]") {
                                    currentCoroutineContext().ensureActive()
                                    emit(piece)
                                    piece = nativeBridge.completionLoop(this@SmolLM, nativePtr)
                                }
                            }
                        } catch (e: IllegalStateException) {
                            throw InferenceFailedException(
                                operation = "SmolLM streaming completion",
                                detail = e.message ?: "The native completion loop failed.",
                                cause = e,
                            )
                        } finally {
                            nativeBridge.stopCompletion(this@SmolLM, nativePtr)
                        }
                    }
                    .flowOn(dispatcher)

    /**
     * Returns the LLM response to the given query as a String. This function is blocking and will
     * return the complete response.
     *
     * @param query The user's query/prompt for the LLM.
     * @param maxTokens Maximum number of tokens to generate. -1 for infinite (until EOS).
     * @param batchSize Number of tokens to generate per JNI call. Values > 1 use batched
     *     generation to reduce JNI boundary crossings. Default is [DEFAULT_BLOCKING_BATCH_SIZE].
     * @return The complete response from the LLM.
     * @throws IllegalStateException if the model is not loaded.
     */
    @JvmOverloads
    fun getResponse(
        query: String,
        maxTokens: Int = -1,
        batchSize: Int = DEFAULT_BLOCKING_BATCH_SIZE,
    ): String {
        verifyHandle()
        logD(LOG_TAG, "getResponse: starting completion. maxTokens=$maxTokens, batchSize=$batchSize, queryLength=${query.length}")
        nativeBridge.startCompletion(this@SmolLM, nativePtr, query)
        try {
            val estimatedCapacity = if (maxTokens > 0) maxTokens * 4 else 512
            val responseBuilder = StringBuilder(estimatedCapacity)
            var tokensGenerated = 0

            if (batchSize > 1) {
                val effectiveBatch = if (maxTokens > 0) minOf(batchSize, maxTokens) else batchSize
                var piece = nativeBridge.completionLoopBatch(this@SmolLM, nativePtr, effectiveBatch)
                while (piece != "[EOG]" && piece.isNotEmpty()) {
                    responseBuilder.append(piece)
                    tokensGenerated += effectiveBatch

                    if (maxTokens > 0 && tokensGenerated >= maxTokens) {
                        logD(LOG_TAG, "getResponse: maxTokens ($maxTokens) reached. Stopping.")
                        break
                    }

                    val remaining =
                        if (maxTokens > 0) minOf(batchSize, maxTokens - tokensGenerated) else batchSize
                    if (remaining <= 0) break
                    piece = nativeBridge.completionLoopBatch(this@SmolLM, nativePtr, remaining)
                }
                if (piece == "[EOG]") {
                    logD(LOG_TAG, "getResponse: [EOG] received after ~$tokensGenerated tokens.")
                }
            } else {
                var piece = nativeBridge.completionLoop(this@SmolLM, nativePtr)
                while (piece != "[EOG]") {
                    responseBuilder.append(piece)
                    tokensGenerated++

                    if (tokensGenerated % 10 == 0) {
                        logD(LOG_TAG, "Generated $tokensGenerated tokens...")
                    }

                    if (maxTokens > 0 && tokensGenerated >= maxTokens) {
                        logD(LOG_TAG, "getResponse: maxTokens ($maxTokens) reached. Stopping.")
                        break
                    }

                    piece = nativeBridge.completionLoop(this@SmolLM, nativePtr)
                }
                if (piece == "[EOG]") {
                    logD(LOG_TAG, "getResponse: [EOG] received after $tokensGenerated tokens.")
                }
            }

            return responseBuilder.toString().also { response ->
                logD(LOG_TAG, "getResponse: finished. Total length=${response.length}")
            }
        } catch (e: IllegalStateException) {
            throw InferenceFailedException(
                operation = "SmolLM completion",
                detail = e.message ?: "The native completion loop failed.",
                cause = e,
            )
        } finally {
            nativeBridge.stopCompletion(this, nativePtr)
        }
    }

    /** Public helper to stop a currently running completion loop (best effort). */
    fun stopCompletion() {
        if (nativePtr == 0L) return
        logD(LOG_TAG, "stopCompletion invoked")
        try {
            nativeBridge.stopCompletion(this, nativePtr)
        } catch (e: Throwable) {
            // best-effort: log and ignore
            logW(LOG_TAG, "stopCompletion failed: ${'$'}{e.message}")
        }
    }

    /**
     * Unloads the LLM model and releases resources. This method should be called when the SmolLM
     * instance is no longer needed to prevent memory leaks.
     */
    override fun close() {
        if (nativePtr != 0L) {
            nativeBridge.close(this, nativePtr)
            nativePtr = 0L
        }
        requestedLoadBackend = null
        selectedBackend = if (useVulkanGPU) ComputeBackend.VULKAN else ComputeBackend.CPU
        currentThinkingMode = ThinkingMode.DEFAULT
        currentReasoningBudget = DEFAULT_REASONING_BUDGET
        loadedInferenceParams = null
    }

    private fun verifyHandle() {
        if (nativePtr == 0L) {
            throw InvalidModelStateException("Model is not loaded. Use SmolLM.load to load the model first.")
        }
    }

    private fun preflightBackendCompatibility(
        modelPath: String,
        params: InferenceParams,
        fileType: Int?,
        dominantTensorType: Int?,
    ) {
        if (!useVulkanGPU || nativeBridge.hasVulkanBackendSupport(this)) {
            return
        }
        val detail =
            buildString {
                append("SmolLM was configured with useVulkan=true, but the active native build does not include Vulkan support")
                append(".")
                append(" nGpuLayers=")
                append(params.nGpuLayers)
                append(", kvCacheTypeK=")
                append(params.kvCacheTypeK.name)
                append(", kvCacheTypeV=")
                append(params.kvCacheTypeV.name)
                append(", ggufFileType=")
                append(describeQuantizedValue(fileType, GGUF_FILE_TYPE_NAMES))
                append(", dominantTensorType=")
                append(describeQuantizedValue(dominantTensorType, GGUF_TENSOR_TYPE_NAMES))
                append(".")
                append(" Disable Vulkan for CPU-only loading or install a Vulkan-enabled llmedge build.")
            }
        throw ModelLoadException(modelPath, detail)
    }

    private fun describeQuantizedValue(value: Int?, names: Map<Int, String>): String =
        when (value) {
            null -> "unknown"
            else -> names[value]?.let { "$it ($value)" } ?: "unknown($value)"
        }

    private external fun loadModel(
            modelPath: String,
            minP: Float,
            temperature: Float,
            storeChats: Boolean,
            contextSize: Long,
            chatTemplate: String,
            nThreads: Int,
            useMmap: Boolean,
            useMlock: Boolean,
            backendId: Int,
            useFlashAttn: Boolean,
            kvCacheTypeK: Int,
            kvCacheTypeV: Int,
            nGpuLayers: Int,
    ): Long

    private external fun setReasoningOptions(
            modelPtr: Long,
            disableThinking: Boolean,
            reasoningBudget: Int,
    )

    private external fun addChatMessage(
            modelPtr: Long,
            message: String,
            role: String,
    )

    private external fun getResponseGenerationSpeed(modelPtr: Long): Float

    private external fun getResponseGeneratedTokenCount(modelPtr: Long): Long

    private external fun getResponseGenerationDurationMicros(modelPtr: Long): Long

    private external fun nativeGetLastGenerationMetrics(modelPtr: Long): LongArray?

    private external fun nativeHasVulkanBackendSupport(): Boolean

    private external fun nativeConfigureThreading(modelPtr: Long, generationThreads: Int, promptThreads: Int)

    private external fun nativeGetEstimatedMemoryBytes(modelPtr: Long): Long

    private external fun nativeGetEstimatedStateMemoryBytes(modelPtr: Long): Long

    private external fun getContextSizeUsed(modelPtr: Long): Int

    // Return native llama_model* pointer for advanced native integrations (do not free)
    private external fun getNativeModelPtr(modelPtr: Long): Long

    /**
     * Public helper to return the underlying native llama_model* pointer. This is intended for
     * advanced integrations (e.g., native projector) and should NOT be used to free or modify the
     * native model directly.
     */
    fun getNativeModelPointer(): Long {
        verifyHandle()
        return nativeBridge.getNativeModelPtr(this, nativePtr)
    }

    // Decode embeddings prepared by the projector (raw floats) without loading mmproj
    private external fun nativeDecodePreparedEmbeddings(
            modelPtr: Long,
            embdPath: String,
            metaPath: String,
            nBatch: Int
    ): Boolean

    // Buffer-based embedding decoding: accepts float array + metadata directly
    private external fun nativeDecodeEmbeddingsBuffer(
            modelPtr: Long,
            embeddings: FloatArray,
            nTokens: Int,
            nx: Int,
            ny: Int,
            embdDim: Int,
            useMrope: Boolean,
            useNonCausal: Boolean,
            nBatch: Int
    ): Boolean
    private external fun nativePrimeImageBuffer(
            modelPtr: Long,
            projectorNativePtr: Long,
            imageData: ByteArray,
            nBatch: Int
    ): Boolean

        // State persistence helpers (KV cache and other context state)
        private external fun nativeGetStateBytes(modelPtr: Long): ByteArray?
        private external fun nativeSetStateBytes(modelPtr: Long, state: ByteArray): Boolean
        private external fun nativeGetSequenceStateBytes(modelPtr: Long, seqId: Int): ByteArray?
        private external fun nativeSetSequenceStateBytes(modelPtr: Long, seqId: Int, state: ByteArray): Boolean
        private external fun nativeClearKvCache(modelPtr: Long)

        private external fun nativeClearMessages(modelPtr: Long)

    /**
     * Decode prepared embeddings previously produced by Projector.encodeImageToFile. This will
     * replay the required llama.decode steps using the current loaded model/context so the image
     * embeddings are present in the KV cache for subsequent generation. Returns true on success.
     */
    fun decodePreparedEmbeddings(embdPath: String, metaPath: String, nBatch: Int = 1): Boolean {
        verifyHandle()
        return try {
            nativeBridge.nativeDecodePreparedEmbeddings(this, nativePtr, embdPath, metaPath, nBatch)
        } catch (e: UnsatisfiedLinkError) {
            false
        }
    }

    /**
     * Decode vision embeddings directly from an in-memory buffer, avoiding temporary file I/O.
     * The embeddings should have been produced by [io.aatricks.llmedge.vision.Projector.encodeImageBuffer].
     * Returns true on success.
     */
    fun decodeEmbeddingsBuffer(embeddings: io.aatricks.llmedge.vision.VisionEmbeddings, nBatch: Int = 1): Boolean {
        verifyHandle()
        return try {
            nativeBridge.nativeDecodeEmbeddingsBuffer(
                this, nativePtr, embeddings.data,
                embeddings.nTokens, embeddings.nx, embeddings.ny,
                embeddings.embdDim, embeddings.useMrope, embeddings.useNonCausal,
                nBatch,
            )
        } catch (e: UnsatisfiedLinkError) {
            false
        }
    }

    internal fun primeImageBuffer(projectorNativePtr: Long, imageData: ByteArray, nBatch: Int = 1): Boolean {
        verifyHandle()
        return try {
            nativeBridge.nativePrimeImageBuffer(this, nativePtr, projectorNativePtr, imageData, nBatch)
        } catch (e: UnsatisfiedLinkError) {
            false
        }
    }

    /**
     * Capture the full model state (including KV cache) as a byte array.
     * Returns null on failure.
     */
    fun getStateBytes(): ByteArray? {
        verifyHandle()
        return try {
            nativeBridge.getStateBytes(this, nativePtr)
        } catch (e: UnsatisfiedLinkError) {
            null
        }
    }

    /**
     * Restore the full model state (including KV cache) from a byte array.
     */
    fun setStateBytes(state: ByteArray): Boolean {
        verifyHandle()
        return try {
            nativeBridge.setStateBytes(this, nativePtr, state)
        } catch (e: UnsatisfiedLinkError) {
            false
        }
    }

    /**
     * Export a single sequence (seq_id) state blob which contains the KV cache for that sequence.
     */
    fun getSequenceStateBytes(seqId: Int = 0): ByteArray? {
        verifyHandle()
        return try {
            nativeBridge.getSequenceStateBytes(this, nativePtr, seqId)
        } catch (e: UnsatisfiedLinkError) {
            null
        }
    }

    /**
     * Import a single sequence (seq_id) state blob which contains the KV cache for that sequence.
     */
    fun setSequenceStateBytes(seqId: Int, state: ByteArray): Boolean {
        verifyHandle()
        return try {
            nativeBridge.setSequenceStateBytes(this, nativePtr, seqId, state)
        } catch (e: UnsatisfiedLinkError) {
            false
        }
    }

    /**
     * Clears the KV cache stored in the model context.
     */
    fun clearKvCache() {
        verifyHandle()
        try {
            nativeBridge.clearKvCache(this, nativePtr)
        } catch (e: UnsatisfiedLinkError) {
            // ignore if not available
        }
    }

    /** Clears any native chat/system messages stored on this runtime. */
    fun clearMessages() {
        verifyHandle()
        try {
            nativeBridge.clearMessages(this, nativePtr)
        } catch (e: UnsatisfiedLinkError) {
            // ignore if not available
        }
    }

    private external fun close(modelPtr: Long)

    private external fun startCompletion(
            modelPtr: Long,
            prompt: String,
    )

    private external fun completionLoop(modelPtr: Long): String

    private external fun completionLoopBatch(modelPtr: Long, maxTokens: Int): String

    private external fun completionLoopBatchBytes(modelPtr: Long, maxTokens: Int): ByteArray?

    private external fun stopCompletion(modelPtr: Long)

    private external fun setThreadAffinity(modelPtr: Long, coreMask: Long)

    private fun applyReasoningState(mode: ThinkingMode, budget: Int) {
        val effectiveMode = if (budget == 0) ThinkingMode.DISABLED else mode
        currentThinkingMode = effectiveMode
        currentReasoningBudget = budget
        if (nativePtr != 0L) {
            nativeBridge.setReasoningOptions(
                    this,
                    nativePtr,
                    effectiveMode.disableReasoning || budget == 0,
                    budget
            )
        }
    }

    private fun resolvedReasoningBudget(mode: ThinkingMode, override: Int?): Int {
        return override ?: if (mode.disableReasoning) 0 else DEFAULT_REASONING_BUDGET
    }

    private fun resolveContextSize(requested: Long?, modelContextSize: Long): Long {
        if (requested != null) {
            // If explicitly requested, trust the caller and clamp only to absolute limits
            return requested.coerceIn(MIN_CONTEXT_SIZE, DEFAULT_CONTEXT_SIZE_CAP)
        }
        val desired = modelContextSize
        val heapAwareCap = recommendedContextCap()
        val effectiveCap = minOf(DEFAULT_CONTEXT_SIZE_CAP, heapAwareCap)
        val clamped = desired.coerceIn(MIN_CONTEXT_SIZE, effectiveCap)
        if (desired != clamped) {
            val heapMb = Runtime.getRuntime().maxMemory() / (1024 * 1024)
            logW(
                    LOG_TAG,
                    "Context window $desired→$clamped tokens to fit heap (${heapMb}MB max). " +
                            "Override via InferenceParams(contextSize=...).",
            )
        }
        return clamped
    }

    private fun resolveChatTemplate(explicit: String?, ggufReader: GGUFReader): String =
            explicit ?: (ggufReader.getChatTemplate() ?: DefaultInferenceParams.chatTemplate)

    private fun recommendedContextCap(): Long {
        val heapMb = Runtime.getRuntime().maxMemory() / (1024 * 1024)
        return when {
            heapMb <= 256 -> 2_048L
            heapMb <= 384 -> 4_096L
            heapMb <= 512 -> 6_144L
            else -> DEFAULT_CONTEXT_SIZE_CAP
        }
    }
}
