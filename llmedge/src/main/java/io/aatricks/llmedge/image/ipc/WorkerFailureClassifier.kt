package io.aatricks.llmedge.image.ipc

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import io.aatricks.llmedge.core.GenerationHangException
import io.aatricks.llmedge.core.WorkerCrashedException
import io.aatricks.llmedge.core.WorkerKilledByMemoryException
import io.aatricks.llmedge.core.WorkerProcessException

/** Maps a dead worker (binderDied) to a typed failure using ApplicationExitInfo (API 30+). */
internal object WorkerFailureClassifier {
    fun classify(
        context: Context,
        pid: Int,
        lastPhase: String,
        lastBackend: String?,
        killedByWatchdog: Boolean,
        stallMs: Long,
    ): WorkerProcessException {
        if (killedByWatchdog) {
            return GenerationHangException(backend = lastBackend, phase = lastPhase, stallMs = stallMs)
        }
        val exitReason = exitReasonFor(context, pid)
        return when (exitReason) {
            ApplicationExitInfo.REASON_LOW_MEMORY -> WorkerKilledByMemoryException()
            else -> WorkerCrashedException(backend = lastBackend, exitReason = exitReason)
        }
    }

    private fun exitReasonFor(
        context: Context,
        pid: Int,
    ): Int? =
        runCatching {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            activityManager.getHistoricalProcessExitReasons(context.packageName, pid, 1)
                .firstOrNull()
                ?.reason
        }.getOrNull()
}
