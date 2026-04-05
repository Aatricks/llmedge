package io.aatricks.llmedge.speech.stt.internal

import android.content.Context
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.core.NativeLibraryCatalog
import io.aatricks.llmedge.core.NativeProbeSupport
import io.aatricks.llmedge.core.runtime.BackendCandidateResolver
import io.aatricks.llmedge.core.runtime.RuntimeLoadPolicy
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.model.ModelFileValidator
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.speech.stt.Whisper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal object WhisperCompanionSupport {
    fun checkBindings(staticInvoker: Whisper): Boolean =
        NativeProbeSupport.unsatisfiedLinkOrDefault(defaultValue = false) {
            staticInvoker.supportCheckBindings()
        }

    fun getVersion(staticInvoker: Whisper): String =
        NativeProbeSupport.unsatisfiedLinkOrDefault(defaultValue = "unknown") {
            staticInvoker.supportGetVersion()
        }

    fun getSystemInfo(staticInvoker: Whisper): String =
        NativeProbeSupport.unsatisfiedLinkOrDefault(defaultValue = "unknown") {
            staticInvoker.supportGetSystemInfo()
        }

    fun getMaxLanguageId(staticInvoker: Whisper): Int =
        NativeProbeSupport.unsatisfiedLinkOrDefault(defaultValue = 0) {
            staticInvoker.supportGetMaxLanguageId()
        }

    fun getLanguageId(
        staticInvoker: Whisper,
        lang: String,
    ): Int =
        NativeProbeSupport.unsatisfiedLinkOrDefault(defaultValue = -1) {
            staticInvoker.supportGetLanguageId(lang)
        }

    fun getLanguageString(
        staticInvoker: Whisper,
        langId: Int,
    ): String =
        NativeProbeSupport.unsatisfiedLinkOrDefault(defaultValue = "unknown") {
            staticInvoker.supportGetLanguageString(langId)
        }

    fun isOpenClAvailable(
        staticInvoker: Whisper,
        overrideValue: Boolean?,
    ): Boolean =
        overrideValue
            ?: NativeProbeSupport.unsatisfiedLinkOrDefault(defaultValue = false) {
                staticInvoker.supportIsOpenClAvailable()
            }

    fun isVulkanBackendAvailable(
        staticInvoker: Whisper,
        overrideValue: Boolean?,
    ): Boolean =
        overrideValue
            ?: NativeProbeSupport.unsatisfiedLinkOrDefault(defaultValue = false) {
                staticInvoker.supportIsVulkanAvailable()
            }

    fun load(
        modelPath: String,
        useGpu: Boolean,
        flashAttn: Boolean,
        gpuDevice: Int,
        staticInvoker: Whisper,
        createHandle: (String, ComputeBackend, Boolean, Int) -> Long,
        openClAvailabilityOverride: Boolean?,
        vulkanAvailabilityOverride: Boolean?,
        onGpuLoadFailure: (ComputeBackend) -> Unit,
    ): Whisper {
        val validatedModel = ModelFileValidator.requireReadableFile(modelPath, "Whisper model")
        val loadRequest =
            BackendCandidateResolver.Request(
                subsystem = ComputeSubsystem.WHISPER,
                allowGpu = useGpu,
                openClAvailable = isOpenClAvailable(staticInvoker, openClAvailabilityOverride),
                vulkanAvailable = isVulkanBackendAvailable(staticInvoker, vulkanAvailabilityOverride),
            )
        val candidates = RuntimeLoadPolicy.candidates(loadRequest)
        var lastError: Throwable? = null
        for (backend in candidates) {
            try {
                val handle =
                    NativeCall.requireHandle(
                        NativeCall.binding(
                            NativeLibraryCatalog.WHISPER_JNI,
                            "Whisper JNI bindings are unavailable.",
                        ) {
                            createHandle(
                                validatedModel.absolutePath,
                                backend,
                                flashAttn,
                                gpuDevice,
                            )
                        },
                        validatedModel.absolutePath,
                        "The native Whisper loader returned an invalid handle.",
                    )
                return Whisper(handle, backend)
            } catch (e: NativeBindingException) {
                throw e
            } catch (e: Throwable) {
                lastError = e
                if (RuntimeLoadPolicy.recordBackendFailureIfNeeded(loadRequest, backend)) {
                    onGpuLoadFailure(backend)
                }
            }
        }

        throw ModelLoadException(
            validatedModel.absolutePath,
            lastError?.message ?: "The native Whisper loader returned an invalid handle.",
            lastError,
        )
    }

    fun loadOnBackend(
        modelPath: String,
        backend: ComputeBackend,
        flashAttn: Boolean,
        gpuDevice: Int,
        createHandle: (String, ComputeBackend, Boolean, Int) -> Long,
    ): Whisper {
        val validatedModel = ModelFileValidator.requireReadableFile(modelPath, "Whisper model")
        val handle =
            NativeCall.requireHandle(
                NativeCall.binding(
                    NativeLibraryCatalog.WHISPER_JNI,
                    "Whisper JNI bindings are unavailable.",
                ) {
                    createHandle(
                        validatedModel.absolutePath,
                        backend,
                        flashAttn,
                        gpuDevice,
                    )
                },
                validatedModel.absolutePath,
                "The native Whisper loader returned an invalid handle.",
            )
        return Whisper(handle, backend)
    }

    suspend fun load(
        context: Context,
        modelPath: String,
        useGpu: Boolean,
        flashAttn: Boolean,
        gpuDevice: Int,
        loadFromPath: (String, Boolean, Boolean, Int) -> Whisper,
    ): Whisper =
        withContext(Dispatchers.IO) {
            val actualPath =
                ModelFileValidator.resolveReadableFile(
                    context,
                    modelPath,
                    "Whisper model",
                ).absolutePath
            loadFromPath(actualPath, useGpu, flashAttn, gpuDevice)
        }

    suspend fun loadFromHuggingFace(
        context: Context,
        modelId: String,
        modelFile: String,
        useGpu: Boolean,
        flashAttn: Boolean,
        gpuDevice: Int,
        token: String?,
        loadFromPath: (String, Boolean, Boolean, Int) -> Whisper,
    ): Whisper =
        withContext(Dispatchers.IO) {
            val result =
                HuggingFaceHub.ensureModelOnDisk(
                    context = context,
                    modelId = modelId,
                    filename = modelFile,
                    token = token,
                )
            loadFromPath(result.file.absolutePath, useGpu, flashAttn, gpuDevice)
        }
}
