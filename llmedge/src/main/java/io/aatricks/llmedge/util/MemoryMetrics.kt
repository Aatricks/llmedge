package io.aatricks.llmedge.util

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.text.format.Formatter

/**
 * Simple helper to snapshot device and process RAM usage.
 *
 * Terminology:
 * - PSS (Proportional Set Size): shared pages are divided among processes that share them,
 *   giving a fairer view of how much RAM a process effectively uses.
 *   It's the recommended single metric for "app RAM usage" on Android.
 *
 * Snapshot fields:
 * - [availSystemMemBytes], [totalSystemMemBytes], [lowMemory]: device-level memory status.
 * - [totalPssKb]: total proportional RAM of the process (KB). Best top-line number to track.
 * - [dalvikPssKb]: managed (Dalvik/ART) heap and runtime structures (KB).
 * - [nativePssKb]: native (C/C++) heap and mmapped resident regions (KB); LLM/ONNX usually show here.
 * - [otherPssKb]: all remaining attributions consolidated by the summary (KB).
 */
object MemoryMetrics {
    data class Snapshot(
        // Device-level
        val availSystemMemBytes: Long,
        val totalSystemMemBytes: Long,
        val lowMemory: Boolean,
        // App process PSS
        val totalPssKb: Int,
        val dalvikPssKb: Int,
        val nativePssKb: Int,
        val otherPssKb: Int,
    ) {
        fun toPretty(context: Context): String {
            fun kb(k: Int) = Formatter.formatShortFileSize(context, k.toLong() * 1024)
            return buildString {
                appendLine("System memory: ${Formatter.formatShortFileSize(context, availSystemMemBytes)} free / ${Formatter.formatShortFileSize(context, totalSystemMemBytes)} total (lowMem=$lowMemory)")
                append("App PSS: total=${kb(totalPssKb)} | dalvik=${kb(dalvikPssKb)} native=${kb(nativePssKb)} other=${kb(otherPssKb)}")
            }
        }
    }

    fun snapshot(context: Context): Snapshot {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)

        val dbg = Debug.MemoryInfo()
        Debug.getMemoryInfo(dbg)

        return Snapshot(
            availSystemMemBytes = memInfo.availMem,
            totalSystemMemBytes = memInfo.totalMem,
            lowMemory = memInfo.lowMemory,
            totalPssKb = dbg.totalPss,
            dalvikPssKb = dbg.dalvikPss,
            nativePssKb = dbg.nativePss,
            otherPssKb = dbg.otherPss,
        )
    }
}
