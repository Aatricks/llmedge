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
import io.aatricks.llmedge.runtime.GGUFReader

import android.content.Context
import io.aatricks.llmedge.core.InvalidModelStateException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.text.runtime.internal.SmolLMCompletionSupport
import io.aatricks.llmedge.text.runtime.internal.SmolLMLoader
import io.aatricks.llmedge.text.runtime.internal.SmolLMRuntimeConfigSupport
import io.aatricks.llmedge.text.runtime.internal.SmolLMStateSupport
import io.aatricks.llmedge.text.runtime.internal.SmolLMVisionInterop
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow

/**
 * Kotlin wrapper for the native LLM runtime. Handles loading models and providing a simple API for
 * running completions and managing model state.
 */
internal fun interface SmolLMNativeLibrarySupport {
    fun ensureLoaded()
}

class SmolLM internal constructor(
    useVulkan: Boolean,
    private val nativeLibrarySupport: SmolLMNativeLibrarySupport,
) : AutoCloseable {
    constructor(useVulkan: Boolean = true) : this(useVulkan, SmolLMCompanionSupport.currentNativeLibrarySupport())

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
        val DEFAULT_BLOCKING_BATCH_SIZE: Int = SmolLMCompanionSupport.defaultBlockingBatchSize

        @JvmStatic
        fun isOpenClAvailable(): Boolean =
            SmolLMCompanionSupport.isOpenClAvailable(::nativeIsOpenClAvailable)

        @JvmStatic
        fun isVulkanBackendAvailable(): Boolean =
            SmolLMCompanionSupport.isVulkanBackendAvailable(::nativeIsVulkanAvailable)

        internal fun overrideNativeBridgeForTests(provider: (SmolLM) -> NativeBridge) {
            SmolLMCompanionSupport.overrideNativeBridgeForTests(provider)
        }

        internal fun resetNativeBridgeForTests() {
            SmolLMCompanionSupport.resetNativeBridgeForTests()
        }

        internal fun overrideNativeLibrarySupportForTests(support: SmolLMNativeLibrarySupport) {
            SmolLMCompanionSupport.overrideNativeLibrarySupportForTests(support)
        }

        internal fun resetNativeLibrarySupportForTests() {
            SmolLMCompanionSupport.resetNativeLibrarySupportForTests()
        }

        internal fun currentNativeLibrarySupport(): SmolLMNativeLibrarySupport =
            SmolLMCompanionSupport.currentNativeLibrarySupport()

        @JvmStatic
        private external fun nativeIsOpenClAvailable(): Boolean

        @JvmStatic
        private external fun nativeIsVulkanAvailable(): Boolean

        internal fun createLoadedForTests(
            nativePtr: Long,
            useVulkan: Boolean = false,
            loadedParams: InferenceParams = InferenceParams(),
        ): SmolLM = SmolLMCompanionSupport.createLoadedForTests(nativePtr, useVulkan, loadedParams)

        internal fun logDebug(message: String) = SmolLMCompanionSupport.logDebug(message)

        internal fun logWarning(message: String) = SmolLMCompanionSupport.logWarning(message)

        internal fun isOpenClBackendAvailable(): Boolean = isOpenClAvailable()

        internal fun isVulkanBackendRuntimeAvailable(): Boolean = isVulkanBackendAvailable()
    }

    init {
        nativeLibrarySupport.ensureLoaded()
    }

    private val runtimeState = SmolLMState(useVulkan, DEFAULT_REASONING_BUDGET)
    private val nativeBridge: NativeBridge = SmolLMCompanionSupport.createNativeBridge(this)
    internal val loadedInferenceParams: InferenceParams?
        get() = runtimeState.loadedInferenceParams

    /** Returns true if this SmolLM instance will try to use Vulkan-backed GPU layers. */
    fun isVulkanEnabled(): Boolean =
        if (runtimeState.nativePtr != 0L) {
            runtimeState.selectedBackend == ComputeBackend.VULKAN
        } else {
            runtimeState.useVulkanGpu
        }

    internal fun getActiveBackend(): ComputeBackend =
        if (runtimeState.nativePtr != 0L) {
            runtimeState.selectedBackend
        } else {
            runtimeState.requestedLoadBackend
                ?: if (runtimeState.useVulkanGpu) ComputeBackend.VULKAN else ComputeBackend.CPU
        }

    internal fun setPreferredBackendForLoad(backend: ComputeBackend?) {
        runtimeState.requestedLoadBackend = backend
    }

    internal fun resolveRequestedBackendForLoad(legacyUseVulkan: Boolean): ComputeBackend =
        runtimeState.requestedLoadBackend
            ?: if (legacyUseVulkan) ComputeBackend.VULKAN else ComputeBackend.CPU

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
    ) = load(modelPath, params, preferredBackend = null)

    internal suspend fun load(
        modelPath: String,
        params: InferenceParams,
        preferredBackend: ComputeBackend?,
    ) = SmolLMLoader.load(this, modelPath, params, preferredBackend)

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
        val nativePtr = requireLoadedHandle()
        bridge.addChatMessage(this, nativePtr, message, "user")
    }

    /** Adds the system prompt for the LLM */
    fun addSystemPrompt(prompt: String) {
        val nativePtr = requireLoadedHandle()
        bridge.addChatMessage(this, nativePtr, prompt, "system")
    }

    /**
     * Adds the assistant message for LLM inference An assistant message is the response given by
     * the LLM for a previous query in the conversation
     */
    fun addAssistantMessage(message: String) {
        val nativePtr = requireLoadedHandle()
        bridge.addChatMessage(this, nativePtr, message, "assistant")
    }

    fun getThinkingMode(): ThinkingMode = runtimeState.currentThinkingMode

    fun getReasoningBudget(): Int = runtimeState.currentReasoningBudget

    fun isThinkingEnabled(): Boolean = runtimeState.currentReasoningBudget != 0

    fun setThinkingMode(mode: ThinkingMode) {
        requireLoadedHandle()
        val targetBudget = if (mode.disableReasoning) 0 else DEFAULT_REASONING_BUDGET
        applyReasoningState(mode, targetBudget)
    }

    fun setThinkingEnabled(enabled: Boolean) {
        setThinkingMode(if (enabled) ThinkingMode.DEFAULT else ThinkingMode.DISABLED)
    }

    fun setReasoningBudget(budget: Int) {
        requireLoadedHandle()
        val mode = if (budget == 0) ThinkingMode.DISABLED else ThinkingMode.DEFAULT
        applyReasoningState(mode, budget)
    }

    /**
     * Returns the rate (in tokens per second) at which the LLM generated its last response via
     * `getResponse()`
     */
    fun getResponseGenerationSpeed(): Float {
        val nativePtr = requireLoadedHandle()
        return bridge.getResponseGenerationSpeed(this, nativePtr)
    }

    /**
     * Returns throughput information for the last completed response. The metrics are reset on the
     * next call to [getResponse] or [getResponseAsFlow].
     */
    fun getLastGenerationMetrics(): GenerationMetrics {
        val nativePtr = requireLoadedHandle()
        return bridge.getLastGenerationMetrics(this, nativePtr)
    }

    fun getEstimatedNativeMemoryBytes(): Long {
        val nativePtr = requireLoadedHandle()
        return bridge.getEstimatedNativeMemoryBytes(this, nativePtr)
    }

    fun getEstimatedStateMemoryBytes(): Long {
        val nativePtr = requireLoadedHandle()
        return bridge.getEstimatedStateMemoryBytes(this, nativePtr)
    }

    /**
     * Returns the number of tokens consumed by the LLM's context window The context of the LLM is
     * roughly the output of, tokenize(apply_chat_template(messages_in_conversation))
     */
    fun getContextLengthUsed(): Int {
        val nativePtr = requireLoadedHandle()
        return bridge.getContextSizeUsed(this, nativePtr)
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
    ): Flow<String> = SmolLMCompletionSupport.getResponseAsFlow(this, query, dispatcher, batchSize)

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
    ): String = SmolLMCompletionSupport.getResponse(this, query, maxTokens, batchSize)

    /** Public helper to stop a currently running completion loop (best effort). */
    fun stopCompletion() {
        SmolLMCompletionSupport.stopCompletion(this)
    }

    /**
     * Unloads the LLM model and releases resources. This method should be called when the SmolLM
     * instance is no longer needed to prevent memory leaks.
     */
    override fun close() {
        val nativePtr = runtimeState.nativePtr
        if (nativePtr != 0L) {
            bridge.close(this, nativePtr)
        }
        runtimeState.reset(DEFAULT_REASONING_BUDGET)
    }

    private fun verifyHandle() {
        if (runtimeState.nativePtr == 0L) {
            throw InvalidModelStateException("Model is not loaded. Use SmolLM.load to load the model first.")
        }
    }

    internal fun requireLoadedHandle(): Long {
        verifyHandle()
        return runtimeState.nativePtr
    }

    internal val bridge: NativeBridge
        get() = nativeBridge

    internal val state: SmolLMState
        get() = runtimeState

    internal fun resolveContextSizeForLoad(requested: Long?, modelContextSize: Long): Long =
        resolveContextSize(requested, modelContextSize)

    internal fun resolveChatTemplateForLoad(explicit: String?, ggufReader: GGUFReader): String =
        resolveChatTemplate(explicit, ggufReader)

    internal fun preflightBackendCompatibilityForLoad(
        modelPath: String,
        params: InferenceParams,
        fileType: Int?,
        dominantTensorType: Int?,
    ) = SmolLMRuntimeConfigSupport.preflightBackendCompatibility(
        useVulkanGpu = runtimeState.useVulkanGpu,
        hasVulkanBackendSupport = bridge.hasVulkanBackendSupport(this),
        modelPath = modelPath,
        params = params,
        fileType = fileType,
        dominantTensorType = dominantTensorType,
        ggufFileTypeNames = GGUF_FILE_TYPE_NAMES,
        ggufTensorTypeNames = GGUF_TENSOR_TYPE_NAMES,
    )

    internal fun applyReasoningStateForLoad(mode: ThinkingMode, budget: Int) =
        SmolLMRuntimeConfigSupport.applyReasoningState(this, mode, budget)

    internal fun resolvedReasoningBudgetForLoad(mode: ThinkingMode, override: Int?): Int =
        SmolLMRuntimeConfigSupport.resolvedReasoningBudget(mode, override, DEFAULT_REASONING_BUDGET)

    internal fun setThreadAffinityForLoad(modelPtr: Long, coreMask: Long) =
        setThreadAffinity(modelPtr, coreMask)

    @JvmName("loadModel")
    internal external fun loadModel(
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

    @JvmName("setReasoningOptions")
    internal external fun setReasoningOptions(
            modelPtr: Long,
            disableThinking: Boolean,
            reasoningBudget: Int,
    )

    @JvmName("addChatMessage")
    internal external fun addChatMessage(
            modelPtr: Long,
            message: String,
            role: String,
    )

    @JvmName("getResponseGenerationSpeed")
    internal external fun getResponseGenerationSpeed(modelPtr: Long): Float

    @JvmName("getResponseGeneratedTokenCount")
    internal external fun getResponseGeneratedTokenCount(modelPtr: Long): Long

    @JvmName("getResponseGenerationDurationMicros")
    internal external fun getResponseGenerationDurationMicros(modelPtr: Long): Long

    @JvmName("nativeGetLastGenerationMetrics")
    internal external fun nativeGetLastGenerationMetrics(modelPtr: Long): LongArray?

    @JvmName("nativeHasVulkanBackendSupport")
    internal external fun nativeHasVulkanBackendSupport(): Boolean

    @JvmName("nativeConfigureThreading")
    internal external fun nativeConfigureThreading(modelPtr: Long, generationThreads: Int, promptThreads: Int)

    @JvmName("nativeGetEstimatedMemoryBytes")
    internal external fun nativeGetEstimatedMemoryBytes(modelPtr: Long): Long

    @JvmName("nativeGetEstimatedStateMemoryBytes")
    internal external fun nativeGetEstimatedStateMemoryBytes(modelPtr: Long): Long

    @JvmName("getContextSizeUsed")
    internal external fun getContextSizeUsed(modelPtr: Long): Int

    // Return native llama_model* pointer for advanced native integrations (do not free)
    @JvmName("getNativeModelPtr")
    internal external fun getNativeModelPtr(modelPtr: Long): Long

    /**
     * Public helper to return the underlying native llama_model* pointer. This is intended for
     * advanced integrations (e.g., native projector) and should NOT be used to free or modify the
     * native model directly.
     */
    fun getNativeModelPointer(): Long {
        val nativePtr = requireLoadedHandle()
        return bridge.getNativeModelPtr(this, nativePtr)
    }

    // Decode embeddings prepared by the projector (raw floats) without loading mmproj
    @JvmName("nativeDecodePreparedEmbeddings")
    internal external fun nativeDecodePreparedEmbeddings(
            modelPtr: Long,
            embdPath: String,
            metaPath: String,
            nBatch: Int
    ): Boolean

    // Buffer-based embedding decoding: accepts float array + metadata directly
    @JvmName("nativeDecodeEmbeddingsBuffer")
    internal external fun nativeDecodeEmbeddingsBuffer(
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
    @JvmName("nativePrimeImageBuffer")
    internal external fun nativePrimeImageBuffer(
            modelPtr: Long,
            projectorNativePtr: Long,
            imageData: ByteArray,
            nBatch: Int
    ): Boolean

    // State persistence helpers (KV cache and other context state)
    @JvmName("nativeGetStateBytes")
    internal external fun nativeGetStateBytes(modelPtr: Long): ByteArray?
    @JvmName("nativeSetStateBytes")
    internal external fun nativeSetStateBytes(modelPtr: Long, state: ByteArray): Boolean
    @JvmName("nativeGetSequenceStateBytes")
    internal external fun nativeGetSequenceStateBytes(modelPtr: Long, seqId: Int): ByteArray?
    @JvmName("nativeSetSequenceStateBytes")
    internal external fun nativeSetSequenceStateBytes(modelPtr: Long, seqId: Int, state: ByteArray): Boolean
    @JvmName("nativeClearKvCache")
    internal external fun nativeClearKvCache(modelPtr: Long)
    @JvmName("nativeClearMessages")
    internal external fun nativeClearMessages(modelPtr: Long)

    /**
     * Decode prepared embeddings previously produced by Projector.encodeImageToFile. This will
     * replay the required llama.decode steps using the current loaded model/context so the image
     * embeddings are present in the KV cache for subsequent generation. Returns true on success.
     */
    fun decodePreparedEmbeddings(embdPath: String, metaPath: String, nBatch: Int = 1): Boolean {
        return SmolLMVisionInterop.decodePreparedEmbeddings(this, embdPath, metaPath, nBatch)
    }

    /**
     * Decode vision embeddings directly from an in-memory buffer, avoiding temporary file I/O.
     * The embeddings should have been produced by [io.aatricks.llmedge.vision.Projector.encodeImageBuffer].
     * Returns true on success.
     */
    fun decodeEmbeddingsBuffer(embeddings: io.aatricks.llmedge.vision.VisionEmbeddings, nBatch: Int = 1): Boolean {
        return SmolLMVisionInterop.decodeEmbeddingsBuffer(this, embeddings, nBatch)
    }

    internal fun primeImageBuffer(projectorNativePtr: Long, imageData: ByteArray, nBatch: Int = 1): Boolean {
        return SmolLMVisionInterop.primeImageBuffer(this, projectorNativePtr, imageData, nBatch)
    }

    /**
     * Capture the full model state (including KV cache) as a byte array.
     * Returns null on failure.
     */
    fun getStateBytes(): ByteArray? {
        return SmolLMStateSupport.getStateBytes(this)
    }

    /**
     * Restore the full model state (including KV cache) from a byte array.
     */
    fun setStateBytes(state: ByteArray): Boolean {
        return SmolLMStateSupport.setStateBytes(this, state)
    }

    /**
     * Export a single sequence (seq_id) state blob which contains the KV cache for that sequence.
     */
    fun getSequenceStateBytes(seqId: Int = 0): ByteArray? {
        return SmolLMStateSupport.getSequenceStateBytes(this, seqId)
    }

    /**
     * Import a single sequence (seq_id) state blob which contains the KV cache for that sequence.
     */
    fun setSequenceStateBytes(seqId: Int, state: ByteArray): Boolean {
        return SmolLMStateSupport.setSequenceStateBytes(this, seqId, state)
    }

    /**
     * Clears the KV cache stored in the model context.
     */
    fun clearKvCache() {
        SmolLMStateSupport.clearKvCache(this)
    }

    /** Clears any native chat/system messages stored on this runtime. */
    fun clearMessages() {
        SmolLMStateSupport.clearMessages(this)
    }

    @JvmName("close")
    internal external fun close(modelPtr: Long)

    @JvmName("startCompletion")
    internal external fun startCompletion(
            modelPtr: Long,
            prompt: String,
    )

    @JvmName("completionLoop")
    internal external fun completionLoop(modelPtr: Long): String

    @JvmName("completionLoopBatch")
    internal external fun completionLoopBatch(modelPtr: Long, maxTokens: Int): String

    @JvmName("completionLoopBatchBytes")
    internal external fun completionLoopBatchBytes(modelPtr: Long, maxTokens: Int): ByteArray?

    @JvmName("stopCompletion")
    internal external fun stopCompletion(modelPtr: Long)

    private external fun setThreadAffinity(modelPtr: Long, coreMask: Long)

    private fun applyReasoningState(mode: ThinkingMode, budget: Int) {
        SmolLMRuntimeConfigSupport.applyReasoningState(this, mode, budget)
    }

    private fun resolvedReasoningBudget(mode: ThinkingMode, override: Int?): Int {
        return SmolLMRuntimeConfigSupport.resolvedReasoningBudget(mode, override, DEFAULT_REASONING_BUDGET)
    }

    private fun resolveContextSize(requested: Long?, modelContextSize: Long): Long {
        return SmolLMRuntimeConfigSupport.resolveContextSize(
            requested = requested,
            modelContextSize = modelContextSize,
            minContextSize = MIN_CONTEXT_SIZE,
            defaultContextSizeCap = DEFAULT_CONTEXT_SIZE_CAP,
        ) { desired, clamped, heapMb ->
            SmolLMCompanionSupport.logWarning(
                "Context window $desired→$clamped tokens to fit heap (${heapMb}MB max). " +
                    "Override via InferenceParams(contextSize=...).",
            )
        }
    }

    private fun resolveChatTemplate(explicit: String?, ggufReader: GGUFReader): String =
            SmolLMRuntimeConfigSupport.resolveChatTemplate(explicit, ggufReader, DefaultInferenceParams.chatTemplate)
}
