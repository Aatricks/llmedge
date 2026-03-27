package io.aatricks.llmedge.text.runtime.internal

import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.model.ModelFileValidator
import io.aatricks.llmedge.runtime.BackendRuntimePolicy
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
    ) = withContext(Dispatchers.IO) {
        val validatedModel = ModelFileValidator.requireGgufFile(modelPath, "SmolLM model")
        if (instance.supportNativePtr != 0L) {
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
                instance.supportResolveContextSize(params.contextSize, modelContextSize)
            resolvedChatTemplate =
                instance.supportResolveChatTemplate(params.chatTemplate, ggufReader)
            fileType = ggufReader.getFileType()
            dominantTensorType = ggufReader.getDominantTensorType()
        } finally {
            ggufReader.close()
        }

        instance.supportPreflightBackendCompatibility(
            modelPath = validatedModel.absolutePath,
            params = params,
            fileType = fileType,
            dominantTensorType = dominantTensorType,
        )

        @Suppress("DEPRECATION")
        val storeChats = params.storeChats
        val promptThreads = params.numThreads.coerceAtLeast(1)
        val backendCandidates =
            instance.supportRequestedLoadBackend?.let(::listOf)
                ?: BackendRuntimePolicy.candidates(
                    subsystem = ComputeSubsystem.TEXT,
                    allowGpu = instance.supportUseVulkanGpu,
                    openClAvailable = SmolLM.supportIsOpenClAvailable(),
                    vulkanAvailable = SmolLM.supportIsVulkanBackendAvailable(),
                )

        var lastLoadError: Throwable? = null
        instance.supportNativePtr = 0L
        for (backend in backendCandidates) {
            instance.supportRequestedLoadBackend = backend
            try {
                val candidateHandle =
                    NativeCall.binding("smollm", "SmolLM JNI bindings are unavailable.") {
                        instance.supportNativeBridge.loadModel(
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
                instance.supportNativePtr =
                    NativeCall.requireHandle(
                        candidateHandle,
                        validatedModel.absolutePath,
                        "The native SmolLM loader returned an invalid handle.",
                    )
                instance.supportSelectedBackend = backend
                break
            } catch (e: NativeBindingException) {
                instance.supportRequestedLoadBackend = null
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
                SmolLM.supportLogW(
                    "Failed to load SmolLM on $backend; retrying with the next backend",
                )
            }
        }
        instance.supportRequestedLoadBackend = null
        if (instance.supportNativePtr == 0L) {
            throw (
                lastLoadError
                    ?: ModelLoadException(
                        validatedModel.absolutePath,
                        "The native SmolLM loader returned an invalid handle.",
                    )
                )
        }

        val generationThreads = (params.generationThreads ?: promptThreads).coerceAtLeast(1)
        instance.supportNativeBridge.configureThreading(
            instance,
            instance.supportNativePtr,
            generationThreads,
            promptThreads,
        )
        val reasoningBudget =
            instance.supportResolvedReasoningBudget(params.thinkingMode, params.reasoningBudget)
        instance.supportApplyReasoningState(params.thinkingMode, reasoningBudget)

        val pCoreMask = CpuTopology.getPerformanceCoreMask()
        if (pCoreMask != 0L) {
            instance.supportSetThreadAffinity(instance.supportNativePtr, pCoreMask)
        }
        instance.supportLoadedInferenceParams =
            params.copy(
                contextSize = resolvedContextSize,
                chatTemplate = resolvedChatTemplate,
                numThreads = promptThreads,
                generationThreads = generationThreads,
            )
    }
}
