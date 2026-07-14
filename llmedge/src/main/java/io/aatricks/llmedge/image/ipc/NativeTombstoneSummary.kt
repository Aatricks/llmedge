package io.aatricks.llmedge.image.ipc

/**
 * Extracts a short, human-readable summary from an Android native-crash tombstone.
 *
 * Since Android 12 the tombstone returned by `ApplicationExitInfo.getTraceInputStream()` is a binary
 * protobuf, but its most diagnostic fields — the signal name, the abort message, and the crashing
 * frames — are stored as plain length-delimited strings. We recover them with a printable-run scan
 * instead of pulling in a protobuf parser, which is enough to tell an assert from a SIGILL and to
 * name the crashing library or op. This lets a field crash on a device we can't reach (no adb) be
 * diagnosed from the app's own error message.
 */
internal object NativeTombstoneSummary {

    private val SIGNAL = Regex("SIG(SEGV|ILL|ABRT|BUS|TRAP|FPE)")
    private val KEYWORDS =
        Regex("(?i)(ggml|GGML_ASSERT|abort message|libsdcpp|stable.?diffusion|vulkan|mul_mat|\\.so\\b)")

    fun summarize(bytes: ByteArray, maxLen: Int = 600): String? {
        val runs = printableRuns(bytes, minLen = 5)
        if (runs.isEmpty()) return null
        val picked = LinkedHashSet<String>()
        // The signal (SIGSEGV / SIGILL / SIGABRT / ...) is the single most diagnostic token.
        runs.firstOrNull { SIGNAL.containsMatchIn(it) }?.let { picked.add(it.take(120)) }
        // Then the abort message and any frames naming our native code or a ggml op.
        for (s in runs) {
            if (picked.size >= 8) break
            if (KEYWORDS.containsMatchIn(s)) picked.add(s.take(160))
        }
        return picked.takeIf { it.isNotEmpty() }?.joinToString(" | ")?.take(maxLen)
    }

    private fun printableRuns(bytes: ByteArray, minLen: Int): List<String> {
        val out = ArrayList<String>()
        val sb = StringBuilder()
        for (b in bytes) {
            val c = b.toInt() and 0xFF
            if (c in 0x20..0x7E) {
                sb.append(c.toChar())
            } else {
                if (sb.length >= minLen) out.add(sb.toString())
                sb.setLength(0)
            }
        }
        if (sb.length >= minLen) out.add(sb.toString())
        return out
    }
}
