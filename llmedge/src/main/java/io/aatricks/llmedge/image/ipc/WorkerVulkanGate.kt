package io.aatricks.llmedge.image.ipc

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.runtime.ComputeBackend

/**
 * Process-wide Vulkan kill switch for the diffusion worker. Some vendor Vulkan drivers (observed:
 * MediaTek mt6855) can SIGSEGV the process merely from being initialized — even when every session
 * computes on CPU — so once Vulkan is disallowed for this worker (by config or by a persisted
 * verdict), `GGML_DISABLE_VULKAN` is set before any native call. That single env var gates the
 * device-capacity probe, ggml's backend registry, and sd.cpp's backend resolution, so the vendor
 * driver is never loaded.
 */
internal object WorkerVulkanGate {
    private const val ENV_VAR = "GGML_DISABLE_VULKAN"
    private const val LOG_TAG = "WorkerVulkanGate"

    /** Any Vulkan verdict disables the driver for the whole worker: it is process-poison, not per-subsystem. */
    fun shouldDisable(
        useVulkan: Boolean,
        blacklistSeed: List<String>,
    ): Boolean = !useVulkan || blacklistSeed.any { it.endsWith(":${ComputeBackend.VULKAN.name}") }

    fun apply(disable: Boolean) {
        runCatching {
            if (disable) {
                android.system.Os.setenv(ENV_VAR, "1", true)
                AndroidLogAdapter.i(LOG_TAG, "Vulkan disabled process-wide for this worker")
            } else {
                android.system.Os.unsetenv(ENV_VAR)
            }
        }.onFailure { error ->
            AndroidLogAdapter.w(LOG_TAG, "Failed to update $ENV_VAR: ${error.message}")
        }
    }
}
