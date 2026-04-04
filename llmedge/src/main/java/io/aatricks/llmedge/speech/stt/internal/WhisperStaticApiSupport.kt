package io.aatricks.llmedge.speech.stt.internal

import android.content.Context
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.speech.stt.Whisper.Companion.LoadBridge
import io.aatricks.llmedge.speech.stt.WhisperRuntimeSupport

internal object WhisperStaticApiSupport {
    const val SAMPLE_RATE: Int = 16000
    const val CHUNK_SIZE_SECONDS: Int = 30

    fun initialize() {
        WhisperRuntimeSupport.ensureNativeLibraryLoaded()
    }

    fun overrideLoadBridgeForTests(provider: () -> LoadBridge) {
        WhisperRuntimeSupport.overrideLoadBridgeForTests(provider)
    }

    fun resetLoadBridgeForTests() {
        WhisperRuntimeSupport.resetLoadBridgeForTests()
    }

    fun overrideBackendAvailabilityForTests(
        openClAvailable: Boolean? = null,
        vulkanAvailable: Boolean? = null,
    ) {
        WhisperRuntimeSupport.overrideBackendAvailabilityForTests(
            openClAvailable = openClAvailable,
            vulkanAvailable = vulkanAvailable,
        )
    }

    fun resetBackendAvailabilityForTests() {
        WhisperRuntimeSupport.resetBackendAvailabilityForTests()
    }

    fun checkBindings(): Boolean =
        WhisperCompanionSupport.checkBindings(WhisperRuntimeSupport.staticInvoker)

    fun getVersion(): String =
        WhisperCompanionSupport.getVersion(WhisperRuntimeSupport.staticInvoker)

    fun getSystemInfo(): String =
        WhisperCompanionSupport.getSystemInfo(WhisperRuntimeSupport.staticInvoker)

    fun getMaxLanguageId(): Int =
        WhisperCompanionSupport.getMaxLanguageId(WhisperRuntimeSupport.staticInvoker)

    fun getLanguageId(lang: String): Int =
        WhisperCompanionSupport.getLanguageId(WhisperRuntimeSupport.staticInvoker, lang)

    fun getLanguageString(langId: Int): String =
        WhisperCompanionSupport.getLanguageString(WhisperRuntimeSupport.staticInvoker, langId)

    fun isOpenClAvailable(): Boolean =
        WhisperCompanionSupport.isOpenClAvailable(
            WhisperRuntimeSupport.staticInvoker,
            WhisperRuntimeSupport.openClAvailabilityOverride(),
        )

    fun isVulkanBackendAvailable(): Boolean =
        WhisperCompanionSupport.isVulkanBackendAvailable(
            WhisperRuntimeSupport.staticInvoker,
            WhisperRuntimeSupport.vulkanAvailabilityOverride(),
        )

    fun load(
        modelPath: String,
        useGpu: Boolean = false,
        flashAttn: Boolean = true,
        gpuDevice: Int = 0,
    ): Whisper =
        WhisperCompanionSupport.load(
            modelPath = modelPath,
            useGpu = useGpu,
            flashAttn = flashAttn,
            gpuDevice = gpuDevice,
            staticInvoker = WhisperRuntimeSupport.staticInvoker,
            createHandle = { path, backend, useFlashAttn, device ->
                WhisperRuntimeSupport.createLoadBridge().create(path, backend, useFlashAttn, device)
            },
            openClAvailabilityOverride = WhisperRuntimeSupport.openClAvailabilityOverride(),
            vulkanAvailabilityOverride = WhisperRuntimeSupport.vulkanAvailabilityOverride(),
            onGpuLoadFailure = WhisperRuntimeSupport::logGpuFallback,
        )

    fun loadOnBackend(
        modelPath: String,
        backend: ComputeBackend,
        flashAttn: Boolean = true,
        gpuDevice: Int = 0,
    ): Whisper =
        WhisperCompanionSupport.loadOnBackend(
            modelPath = modelPath,
            backend = backend,
            flashAttn = flashAttn,
            gpuDevice = gpuDevice,
        ) { path, chosenBackend, useFlashAttn, device ->
            WhisperRuntimeSupport.createLoadBridge().create(path, chosenBackend, useFlashAttn, device)
        }

    suspend fun load(
        context: Context,
        modelPath: String,
        useGpu: Boolean = false,
        flashAttn: Boolean = true,
        gpuDevice: Int = 0,
    ): Whisper =
        WhisperCompanionSupport.load(
            context = context,
            modelPath = modelPath,
            useGpu = useGpu,
            flashAttn = flashAttn,
            gpuDevice = gpuDevice,
            loadFromPath = ::load,
        )

    suspend fun loadFromHuggingFace(
        context: Context,
        modelId: String = "ggerganov/whisper.cpp",
        modelFile: String = "ggml-base.bin",
        useGpu: Boolean = false,
        flashAttn: Boolean = true,
        gpuDevice: Int = 0,
        token: String? = null,
    ): Whisper =
        WhisperCompanionSupport.loadFromHuggingFace(
            context = context,
            modelId = modelId,
            modelFile = modelFile,
            useGpu = useGpu,
            flashAttn = flashAttn,
            gpuDevice = gpuDevice,
            token = token,
            loadFromPath = ::load,
        )
}
