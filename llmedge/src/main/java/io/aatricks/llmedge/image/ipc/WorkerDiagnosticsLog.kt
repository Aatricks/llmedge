package io.aatricks.llmedge.image.ipc

import android.content.Context
import android.os.SystemClock
import io.aatricks.llmedge.core.AndroidLogAdapter
import java.io.File

/**
 * Mirrors the worker process's own [AndroidLogAdapter] output to a file in the shared data
 * directory, so the trail leading up to a crash survives the process.
 *
 * The worker's diagnostics — which pass acquired which backend, which runtime was invalidated
 * before which phase — otherwise exist only in logcat, which a field reporter without a PC cannot
 * read. The host reads the tail of this file after `binderDied` and folds it into
 * [io.aatricks.llmedge.core.WorkerCrashedException], which is what a shared app log ends up
 * containing.
 *
 * Lines are flushed per write: a buffered line is a line lost to the crash it was meant to
 * explain.
 */
internal object WorkerDiagnosticsLog {
    private const val MAX_BYTES = 256 * 1024
    private const val MAX_LINES = 400
    private const val STALE_FILE_AGE_MS = 10 * 60 * 1000L

    private val recent = ArrayDeque<String>()
    private var file: File? = null

    fun logFile(context: Context, pid: Int): File =
        File(context.filesDir, "diffusion-worker-log-$pid.txt")

    /** Installs the sink for this process. Call once, before any generation work. */
    @Synchronized
    fun install(context: Context, pid: Int) {
        val target = logFile(context, pid)
        runCatching { target.delete() }
        file = target
        recent.clear()
        sweepStaleFiles(context, pid)
        AndroidLogAdapter.setSink { level, tag, message, throwable ->
            append("$level/$tag: $message" + (throwable?.let { "\n${it.stackTraceToString()}" } ?: ""))
        }
    }

    @Synchronized
    private fun append(line: String) {
        val target = file ?: return
        val stamped = "[${SystemClock.uptimeMillis()}] $line"
        recent.addLast(stamped)
        while (recent.size > MAX_LINES) {
            recent.removeFirst()
        }
        runCatching {
            if (target.length() > MAX_BYTES) {
                target.writeText(recent.joinToString("\n", postfix = "\n"))
            } else {
                target.appendText("$stamped\n")
            }
        }
    }

    /** Reads and removes the tail left behind by a dead worker. */
    fun consumeTail(context: Context, pid: Int, maxChars: Int): String? =
        runCatching {
            val target = logFile(context, pid)
            if (!target.isFile) return null
            val text = target.readText().trim()
            target.delete()
            text.takeLast(maxChars).ifBlank { null }
        }.getOrNull()

    /** Drops logs left by workers that died without the host ever reading them. */
    private fun sweepStaleFiles(context: Context, currentPid: Int) {
        runCatching {
            val cutoff = System.currentTimeMillis() - STALE_FILE_AGE_MS
            context.filesDir
                .listFiles { candidate ->
                    candidate.name.startsWith("diffusion-worker-log-") &&
                        candidate.name != "diffusion-worker-log-$currentPid.txt" &&
                        candidate.lastModified() < cutoff
                }
                .orEmpty()
                .forEach(File::delete)
        }
    }
}
