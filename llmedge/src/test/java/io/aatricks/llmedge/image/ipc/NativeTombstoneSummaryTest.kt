package io.aatricks.llmedge.image.ipc

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeTombstoneSummaryTest {
    /** Builds a binary blob interleaving printable tokens with non-printable separators. */
    private fun tombstone(vararg tokens: String): ByteArray {
        val out = ArrayList<Byte>()
        for (t in tokens) {
            out.add(0); out.add(1) // non-printable separator so tokens are distinct runs
            for (c in t) out.add(c.code.toByte())
        }
        out.add(0)
        return out.toByteArray()
    }

    @Test
    fun `recovers signal, abort message and crashing library from a protobuf tombstone`() {
        val bytes = tombstone(
            "random-noise",
            "SIGABRT",
            "Abort message: 'GGML_ASSERT: ggml.c:5423: ggml_nbytes(a) failed'",
            "libsdcpp.so",
            "some_unrelated_symbol",
        )

        val summary = NativeTombstoneSummary.summarize(bytes)

        assertNotNull(summary)
        assertTrue("signal", summary!!.contains("SIGABRT"))
        assertTrue("abort message", summary.contains("GGML_ASSERT"))
        assertTrue("crashing library", summary.contains("libsdcpp.so"))
    }

    @Test
    fun `recovers SIGILL for an illegal-instruction crash`() {
        val summary = NativeTombstoneSummary.summarize(tombstone("SIGILL", "mul_mat"))
        assertNotNull(summary)
        assertTrue(summary!!.contains("SIGILL"))
    }

    @Test
    fun `returns null when there is nothing readable`() {
        assertNull(NativeTombstoneSummary.summarize(byteArrayOf(0, 1, 2, 3, 4, 0, 1)))
    }

    @Test
    fun `ignores readable strings that are not crash-relevant`() {
        // Only noise words, none matching a signal or keyword -> no useful summary.
        assertNull(NativeTombstoneSummary.summarize(tombstone("hello", "world", "abcdef")))
    }
}
