package io.aatricks.llmedge.image.diffusion

import android.content.Context
import android.os.Build
import java.io.File

/**
 * Cross-process breadcrumb written when a Vulkan `nativeCreate` attempt fails and the loader falls
 * back to CPU. The load runs inside the worker process, which cannot safely share the host's
 * verdict SharedPreferences, so the failure is recorded as a file (the same pattern as the worker
 * crash breadcrumb) and converted into a persisted backend verdict by the host engine. Keyed to
 * [Build.FINGERPRINT] so an OS/driver update discards the marker.
 */
internal object VulkanCreateFailureMarker {
    fun file(context: Context): File = File(context.filesDir, "diffusion-vulkan-create-failed")

    fun record(context: Context) {
        runCatching { file(context).writeText(Build.FINGERPRINT) }
    }

    /** True when a marker from the current OS build exists; the marker is deleted either way. */
    fun consume(context: Context): Boolean =
        runCatching {
            val marker = file(context)
            if (!marker.isFile) return false
            val fingerprint = marker.readText().trim()
            marker.delete()
            fingerprint == Build.FINGERPRINT
        }.getOrDefault(false)
}
