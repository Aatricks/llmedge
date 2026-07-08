package io.aatricks.llmedge.image.ipc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ProcCpuReaderTest {
    @Test
    fun `parses utime plus stime from a standard stat line`() {
        // Fields 14 (utime) and 15 (stime) are 500 and 250 jiffies.
        val stat = "1234 (llmedge_sd) S 1 1234 0 0 -1 4194560 100 0 0 0 500 250 0 0 20 0 30 0 12345 100000 200 18446744073709551615"
        // 750 jiffies at 100 Hz = 7500 ms
        assertEquals(7500L, ProcCpuReader.parseStat(stat, clockTicksPerSecond = 100))
    }

    @Test
    fun `handles comm fields containing spaces and parens`() {
        val stat = "42 (weird (name) x) R 1 42 0 0 -1 0 0 0 0 0 100 100 0 0 20 0 1 0 1 1 1 1"
        assertEquals(2000L, ProcCpuReader.parseStat(stat, clockTicksPerSecond = 100))
    }

    @Test
    fun `returns null for blank or malformed input`() {
        assertNull(ProcCpuReader.parseStat(null))
        assertNull(ProcCpuReader.parseStat(""))
        assertNull(ProcCpuReader.parseStat("12 (x) S 1 2"))
    }
}
