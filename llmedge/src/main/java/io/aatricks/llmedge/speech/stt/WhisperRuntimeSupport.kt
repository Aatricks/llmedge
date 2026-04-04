package io.aatricks.llmedge.speech.stt

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.NativeBridgeProvider
import io.aatricks.llmedge.core.NativeLibraryLoader
import io.aatricks.llmedge.runtime.ComputeBackend

internal object WhisperRuntimeSupport {
    private const val LOG_TAG = "Whisper"
    const val SAMPLE_RATE: Int = 16000
    const val CHUNK_SIZE_SECONDS: Int = 30

    internal val staticInvoker by lazy { Whisper(0L, ComputeBackend.CPU) }

    private val loadBridgeProvider =
        NativeBridgeProvider<Unit, Whisper.Companion.LoadBridge> { _ ->
            object : Whisper.Companion.LoadBridge {
                override fun create(
                    modelPath: String,
                    backend: ComputeBackend,
                    flashAttn: Boolean,
                    gpuDevice: Int,
                ): Long =
                    staticInvoker.nativeCreate(
                        modelPath,
                        backend.id,
                        flashAttn,
                        gpuDevice,
                    )
            }
        }

    private var openClAvailabilityOverrideForTests: Boolean? = null
    private var vulkanAvailabilityOverrideForTests: Boolean? = null

    fun ensureNativeLibraryLoaded() {
        NativeLibraryLoader.ensureWhisperLoaded(
            required = false,
            onDebug = { message -> AndroidLogAdapter.d(LOG_TAG, message) },
            onError = { message, throwable -> AndroidLogAdapter.e(LOG_TAG, message, throwable) },
        )
    }

    fun createLoadBridge(): Whisper.Companion.LoadBridge = loadBridgeProvider.create(Unit)

    fun overrideLoadBridgeForTests(provider: () -> Whisper.Companion.LoadBridge) {
        loadBridgeProvider.override { _ -> provider() }
    }

    fun resetLoadBridgeForTests() {
        loadBridgeProvider.reset()
    }

    fun overrideBackendAvailabilityForTests(
        openClAvailable: Boolean? = null,
        vulkanAvailable: Boolean? = null,
    ) {
        openClAvailabilityOverrideForTests = openClAvailable
        vulkanAvailabilityOverrideForTests = vulkanAvailable
    }

    fun resetBackendAvailabilityForTests() {
        openClAvailabilityOverrideForTests = null
        vulkanAvailabilityOverrideForTests = null
    }

    fun openClAvailabilityOverride(): Boolean? = openClAvailabilityOverrideForTests

    fun vulkanAvailabilityOverride(): Boolean? = vulkanAvailabilityOverrideForTests

    fun logGpuFallback(backend: ComputeBackend) {
        AndroidLogAdapter.w(LOG_TAG, "Failed to load Whisper on $backend; retrying with the next backend")
    }
}
