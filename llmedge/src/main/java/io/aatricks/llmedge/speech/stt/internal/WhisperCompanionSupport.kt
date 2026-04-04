package io.aatricks.llmedge.speech.stt.internal

import android.content.Context
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeCall
import io.aatricks.llmedge.core.NativeLibraryCatalog
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
        try {
            staticInvoker.supportCheckBindings()
        } catch (_: UnsatisfiedLinkError) {
            false
        }

    fun getVersion(staticInvoker: Whisper): String =
        try {
            staticInvoker.supportGetVersion()
        } catch (_: UnsatisfiedLinkError) {
            "unknown"
        }

    fun getSystemInfo(staticInvoker: Whisper): String =
        try {
            staticInvoker.supportGetSystemInfo()
        } catch (_: UnsatisfiedLinkError) {
            "unknown"
        }

    fun getMaxLanguageId(staticInvoker: Whisper): Int =
        try {
            staticInvoker.supportGetMaxLanguageId()
        } catch (_: UnsatisfiedLinkError) {
            0
        }

    fun getLanguageId(
        staticInvoker: Whisper,
        lang: String,
    ): Int =
        try {
            staticInvoker.supportGetLanguageId(lang)
        } catch (_: UnsatisfiedLinkError) {
            -1
        }

    fun getLanguageString(
        staticInvoker: Whisper,
        langId: Int,
    ): String =
        try {
            staticInvoker.supportGetLanguageString(langId)
        } catch (_: UnsatisfiedLinkError) {
            "unknown"
        }

    fun isOpenClAvailable(
        staticInvoker: Whisper,
        overrideValue: Boolean?,
    ): Boolean =
        overrideValue
            ?:
            try {
                staticInvoker.supportIsOpenClAvailable()
            } catch (_: UnsatisfiedLinkError) {
                false
            }

    fun isVulkanBackendAvailable(
        staticInvoker: Whisper,
        overrideValue: Boolean?,
    ): Boolean =
        overrideValue
            ?:
            try {
                staticInvoker.supportIsVulkanAvailable()
            } catch (_: UnsatisfiedLinkError) {
                false
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
