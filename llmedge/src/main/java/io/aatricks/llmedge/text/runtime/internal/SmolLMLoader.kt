package io.aatricks.llmedge.text.runtime.internal

import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.core.NativeLibraryCatalog
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.RuntimeLoadPolicy
import io.aatricks.llmedge.core.runtime.runBackendAttempts
import io.aatricks.llmedge.model.ModelFileValidator
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal object SmolLMLoader {
    suspend fun load(
        instance: SmolLM,
        modelPath: String,
        params: SmolLM.InferenceParams,
        preferredBackend: ComputeBackend?,
    ) = withContext(Dispatchers.IO) {
        val validatedModel = ModelFileValidator.requireGgufFile(modelPath, "SmolLM model")
        if (instance.state.nativePtr != 0L) {
            instance.close()
        }

        val ggufReader = GGUFReader()
        val resolvedContextSize: Long
        val resolvedChatTemplate: String
        val fileType: Int?
        val dominantTensorType: Int?
        try {
            ggufReader.load(validatedModel.absolutePath)
            val modelContextSize =
                ggufReader.getContextSize() ?: SmolLM.DefaultInferenceParams.contextSize
            resolvedContextSize =
                instance.resolveContextSizeForLoad(params.contextSize, modelContextSize)
            resolvedChatTemplate =
                instance.resolveChatTemplateForLoad(params.chatTemplate, ggufReader)
            fileType = ggufReader.getFileType()
            dominantTensorType = ggufReader.getDominantTensorType()
        } finally {
            ggufReader.close()
        }

        instance.preflightBackendCompatibilityForLoad(
            modelPath = validatedModel.absolutePath,
            params = params,
            fileType = fileType,
            dominantTensorType = dominantTensorType,
        )

        @Suppress("DEPRECATION")
        val storeChats = params.storeChats
        val promptThreads = params.numThreads.coerceAtLeast(1)
        val loadRequest =
            BackendCandidateResolver.Request(
                subsystem = ComputeSubsystem.TEXT,
                allowGpu = instance.state.useVulkanGpu,
                openClAvailable = SmolLM.isOpenClBackendAvailable(),
                vulkanAvailable = SmolLM.isVulkanBackendRuntimeAvailable(),
            )
        val backendCandidates =
            RuntimeLoadPolicy.candidates(
                request = loadRequest,
                preferredBackend = preferredBackend ?: instance.state.requestedLoadBackend,
                includeCpuFallback = preferredBackend == null && instance.state.requestedLoadBackend == null,
            )

        instance.state.nativePtr = 0L
        try {
            val selectedBackend =
                runBackendAttempts(
                    candidates = backendCandidates,
                    onFailure = { backend, error ->
                        if (RuntimeLoadPolicy.recordBackendFailureIfNeeded(loadRequest, backend, preferredBackend)) {
                            val detail = error?.message?.let { ": $it" } ?: ""
                            SmolLM.logWarning("Failed to load SmolLM on $backend; retrying with the next backend$detail")
                        }
                    },
                    exhaustedError = { lastLoadError ->
                        lastLoadError
                            ?: ModelLoadException(
                                validatedModel.absolutePath,
                                "The native SmolLM loader returned an invalid handle.",
                            )
                    },
                ) { backend ->
                    // Q8_KV KV-cache kernels exist only in the CPU (IQK) path of the ik fork;
                    // loading it on a GPU backend SIGSEGVs natively (measured on S22 Vulkan).
                    // Failing the attempt here lets runBackendAttempts fall back to CPU.
                    if (backend != ComputeBackend.CPU &&
                        params.kvCacheTypeK == SmolLM.KvCacheType.Q8_KV
                    ) {
                        throw ModelLoadException(
                            validatedModel.absolutePath,
                            "KvCacheType.Q8_KV is CPU-only; refusing to load it on $backend.",
                        )
                    }
                    instance.state.requestedLoadBackend = backend
                    try {
                        val candidateHandle =
                            NativeCall.binding(
                                NativeLibraryCatalog.SMOLLM,
                                "SmolLM JNI bindings are unavailable.",
                            ) {
                                instance.bridge.loadModel(
                                    instance,
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
                                    params.nUbatch,
                                )
                            }
                        instance.state.nativePtr =
                            NativeCall.requireHandle(
                                candidateHandle,
                                validatedModel.absolutePath,
                                "The native SmolLM loader returned an invalid handle.",
                            )
                        backend
                    } catch (error: IllegalStateException) {
                        throw ModelLoadException(
                            validatedModel.absolutePath,
                            error.message ?: "The native SmolLM loader reported an unknown error.",
                            error,
                        )
                    }
                } ?: error("SmolLM backend attempts exhausted without an error")
            instance.state.selectedBackend = selectedBackend
        } finally {
            instance.state.requestedLoadBackend = null
        }

        val generationThreads = (params.generationThreads ?: promptThreads).coerceAtLeast(1)
        instance.bridge.configureThreading(
            instance,
            instance.state.nativePtr,
            generationThreads,
            promptThreads,
        )
        val reasoningBudget =
            instance.resolvedReasoningBudgetForLoad(params.thinkingMode, params.reasoningBudget)
        instance.applyReasoningStateForLoad(params.thinkingMode, reasoningBudget)

        val pCoreMask = CpuTopology.getPerformanceCoreMask()
        instance.state.pCoreMask = pCoreMask
        if (pCoreMask != 0L) {
            instance.setThreadAffinityForLoad(instance.state.nativePtr, pCoreMask)
        }
        instance.state.loadedInferenceParams =
            params.copy(
                contextSize = resolvedContextSize,
                chatTemplate = resolvedChatTemplate,
                numThreads = promptThreads,
                generationThreads = generationThreads,
            )
    }
}
