package io.aatricks.llmedge.image.ipc

import android.content.Context
import android.os.Build
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem

/**
 * Persisted backend hang verdicts, host process only (the worker receives a seed over IPC).
 * Keyed by [Build.FINGERPRINT]: an OS/driver update clears all verdicts, so a fixed GPU driver
 * (e.g. the Pixel 10 QPR3 PowerVR bump) automatically re-enables Vulkan.
 */
internal class BackendVerdictStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun recordHang(
        subsystem: ComputeSubsystem,
        backend: ComputeBackend,
    ) {
        if (backend == ComputeBackend.CPU) return
        prefs.edit()
            .putString(KEY_FINGERPRINT, Build.FINGERPRINT)
            .putLong(verdictKey(subsystem, backend), System.currentTimeMillis())
            .apply()
    }

    /** Verdicts for the current OS fingerprint; clears stale ones from before an OS update. */
    fun load(): List<Pair<ComputeSubsystem, ComputeBackend>> {
        val stored = prefs.getString(KEY_FINGERPRINT, null) ?: return emptyList()
        if (stored != Build.FINGERPRINT) {
            reset()
            return emptyList()
        }
        return prefs.all.keys
            .filter { it.startsWith(VERDICT_PREFIX) }
            .mapNotNull { key ->
                val parts = key.removePrefix(VERDICT_PREFIX).split('.')
                if (parts.size != 2) return@mapNotNull null
                val subsystem = ComputeSubsystem.entries.firstOrNull { it.name == parts[0] } ?: return@mapNotNull null
                val backend = ComputeBackend.entries.firstOrNull { it.name == parts[1] } ?: return@mapNotNull null
                subsystem to backend
            }
    }

    fun reset() {
        prefs.edit().clear().apply()
    }

    /** [availability.vulkanDeviceInfo.freeMemoryMB] is a probe-time snapshot. */
    fun recordImageProbe(availability: io.aatricks.llmedge.ComputeBackendAvailability) {
        val info = availability.vulkanDeviceInfo
        prefs.edit()
            .putString(KEY_FINGERPRINT, Build.FINGERPRINT)
            .putBoolean("probe.image.opencl", availability.openClAvailable)
            .putBoolean("probe.image.vulkan", availability.vulkanAvailable)
            .putInt("probe.image.vkDeviceCount", info?.deviceCount ?: 0)
            .putLong("probe.image.vkFreeMb", info?.freeMemoryMB ?: 0L)
            .putLong("probe.image.vkTotalMb", info?.totalMemoryMB ?: 0L)
            .apply()
    }

    fun loadImageProbe(): io.aatricks.llmedge.ComputeBackendAvailability? {
        val stored = prefs.getString(KEY_FINGERPRINT, null) ?: return null
        if (stored != Build.FINGERPRINT) {
            reset()
            return null
        }
        if (!prefs.contains("probe.image.vulkan")) return null
        val vkDeviceCount = prefs.getInt("probe.image.vkDeviceCount", 0)
        val info = if (vkDeviceCount > 0) {
            io.aatricks.llmedge.VulkanDeviceInfo(
                deviceCount = vkDeviceCount,
                freeMemoryMB = prefs.getLong("probe.image.vkFreeMb", 0L),
                totalMemoryMB = prefs.getLong("probe.image.vkTotalMb", 0L),
                deviceIndex = 0
            )
        } else null
        return io.aatricks.llmedge.ComputeBackendAvailability(
            openClAvailable = prefs.getBoolean("probe.image.opencl", false),
            vulkanAvailable = prefs.getBoolean("probe.image.vulkan", false),
            vulkanDeviceInfo = info
        )
    }

    private fun verdictKey(
        subsystem: ComputeSubsystem,
        backend: ComputeBackend,
    ): String = "$VERDICT_PREFIX${subsystem.name}.${backend.name}"

    companion object {
        private const val PREFS_NAME = "llmedge_backend_verdicts"
        private const val KEY_FINGERPRINT = "fingerprint"
        private const val VERDICT_PREFIX = "verdict."
    }
}
