package io.aatricks.llmedge.text.runtime.internal

import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.core.NativeLibraryCatalog
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
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
        val backendCandidates =
            preferredBackend?.let(::listOf)
                ?: instance.state.requestedLoadBackend?.let(::listOf)
                ?: BackendCandidateResolver.candidates(
                    BackendCandidateResolver.Request(
                        subsystem = ComputeSubsystem.TEXT,
                        allowGpu = instance.state.useVulkanGpu,
                        openClAvailable = SmolLM.isOpenClBackendAvailable(),
                        vulkanAvailable = SmolLM.isVulkanBackendRuntimeAvailable(),
                    ),
                )

        var lastLoadError: Throwable? = null
        instance.state.nativePtr = 0L
        for (backend in backendCandidates) {
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
                        )
                    }
                instance.state.nativePtr =
                    NativeCall.requireHandle(
                        candidateHandle,
                        validatedModel.absolutePath,
                        "The native SmolLM loader returned an invalid handle.",
                    )
                instance.state.selectedBackend = backend
                break
            } catch (e: NativeBindingException) {
                instance.state.requestedLoadBackend = null
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

            if (backend != ComputeBackend.CPU && preferredBackend == null) {
                BackendCandidateResolver.blacklist(ComputeSubsystem.TEXT, backend)
                SmolLM.logWarning(
                    "Failed to load SmolLM on $backend; retrying with the next backend",
                )
            }
        }
        instance.state.requestedLoadBackend = null
        if (instance.state.nativePtr == 0L) {
            throw (
                lastLoadError
                    ?: ModelLoadException(
                        validatedModel.absolutePath,
                        "The native SmolLM loader returned an invalid handle.",
                    )
                )
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
