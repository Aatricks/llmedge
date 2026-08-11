package io.aatricks.llmedge.image.ipc

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import io.aatricks.llmedge.core.GenerationHangException
import io.aatricks.llmedge.core.WorkerCrashedException
import io.aatricks.llmedge.core.WorkerKilledByMemoryException
import io.aatricks.llmedge.core.WorkerProcessException
import io.aatricks.llmedge.runtime.ComputeBackend
import java.io.File

/** Maps a dead worker (binderDied) to a typed failure using ApplicationExitInfo (API 30+). */
internal object WorkerFailureClassifier {
    /** Kept generous: this is the whole post-mortem when no tombstone or breadcrumb exists. */
    private const val WORKER_LOG_TAIL_CHARS = 4000

    /** Breadcrumb the worker writes on an uncaught JVM exception (see [DiffusionWorkerService]). */
    internal fun crashBreadcrumbFile(context: Context, pid: Int): File =
        File(context.filesDir, "diffusion-worker-crash-$pid.txt")

    fun classify(
        context: Context,
        pid: Int,
        lastPhase: String,
        lastBackend: String?,
        killedByWatchdog: Boolean,
        stallMs: Long,
        hardWall: Boolean = false,
    ): WorkerProcessException {
        if (killedByWatchdog) {
            return GenerationHangException(
                backend = lastBackend,
                phase = lastPhase,
                stallMs = stallMs,
                hardWall = hardWall,
            )
        }
        val exitInfo = exitInfoFor(context, pid)
        return when (exitInfo?.reason) {
            ApplicationExitInfo.REASON_LOW_MEMORY -> WorkerKilledByMemoryException()
            else -> WorkerCrashedException(
                backend = lastBackend,
                exitReason = exitInfo?.reason,
                crashSummary = buildCrashSummary(context, exitInfo, pid, lastPhase),
            )
        }
    }

    private fun exitInfoFor(context: Context, pid: Int): ApplicationExitInfo? =
        runCatching {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            activityManager.getHistoricalProcessExitReasons(context.packageName, pid, 1)
                .firstOrNull()
        }.getOrNull()

    /**
     * Recovers a diagnosable post-mortem without adb: the native tombstone's abort message + signal
     * for a native crash, or the worker's persisted uncaught-exception stack for a JVM crash, always
     * prefixed with the phase the generation reached.
     */
    private fun buildCrashSummary(
        context: Context,
        exitInfo: ApplicationExitInfo?,
        pid: Int,
        lastPhase: String,
    ): String {
        val parts = mutableListOf("phase=$lastPhase")

        if (exitInfo != null &&
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
            exitInfo.reason == ApplicationExitInfo.REASON_CRASH_NATIVE
        ) {
            runCatching { exitInfo.traceInputStream?.use { it.readBytes() } }
                .getOrNull()
                ?.let { NativeTombstoneSummary.summarize(it) }
                ?.let { parts.add(it) }
        }

        readCrashBreadcrumb(context, pid)?.let { parts.add(it) }

        // Distinguish a CPU crash that was a *forced* fallback (a GPU backend was blacklisted after
        // a prior hang on this OS build) from one on a device that never had a usable GPU — the key
        // question when a GPU-capable device dies on the CPU path.
        runCatching { BackendVerdictStore(context).load() }
            .getOrNull()
            ?.firstOrNull { it.second != ComputeBackend.CPU }
            ?.let { parts.add("gpu-disabled-after-prior-hang(${it.first.name}:${it.second.name})") }

        if (parts.size == 1) {
            exitInfo?.description?.takeIf { it.isNotBlank() }?.let { parts.add(it) }
        }

        // Always last, and always present: when none of the above resolved a cause, the worker's
        // own log trail is what tells us which pass it died in.
        WorkerDiagnosticsLog.consumeTail(context, pid, WORKER_LOG_TAIL_CHARS)
            ?.let { parts.add("worker-log:\n$it") }

        return parts.joinToString("; ")
    }

    private fun readCrashBreadcrumb(context: Context, pid: Int): String? =
        runCatching {
            val file = crashBreadcrumbFile(context, pid)
            if (!file.isFile) return null
            val text = file.readText().trim().take(1000)
            file.delete()
            text.ifBlank { null }
        }.getOrNull()
}
