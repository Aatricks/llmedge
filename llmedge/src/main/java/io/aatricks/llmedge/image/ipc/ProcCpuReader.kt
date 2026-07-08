package io.aatricks.llmedge.image.ipc

import android.system.Os
import android.system.OsConstants
import java.io.File

/**
 * Reads a same-UID process's cumulative CPU time from /proc/<pid>/stat (utime + stime).
 * Used by the watchdog to tell a driver deadlock (flat CPU) from a long-but-live workload
 * such as a cold Vulkan shader compile (pegs a core).
 */
internal object ProcCpuReader {
    /** Cumulative user+system CPU time in ms, or null when /proc is unreadable. */
    fun readCpuTimeMs(pid: Int): Long? = parseStat(runCatching { File("/proc/$pid/stat").readText() }.getOrNull())

    fun parseStat(
        stat: String?,
        clockTicksPerSecond: Long = clkTck,
    ): Long? {
        if (stat.isNullOrBlank()) return null
        // The comm field (2nd, in parens) may contain spaces; fields resume after the last ')'.
        val afterComm = stat.substringAfterLast(')').trim()
        val fields = afterComm.split(' ')
        // afterComm starts at field 3 ("state"); utime/stime are fields 14/15 of the full line.
        val utime = fields.getOrNull(11)?.toLongOrNull() ?: return null
        val stime = fields.getOrNull(12)?.toLongOrNull() ?: return null
        if (clockTicksPerSecond <= 0) return null
        return (utime + stime) * 1000L / clockTicksPerSecond
    }

    private val clkTck: Long by lazy {
        runCatching { Os.sysconf(OsConstants._SC_CLK_TCK) }.getOrDefault(100L).takeIf { it > 0 } ?: 100L
    }
}
