package io.aatricks.llmedge.huggingface

import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class HFDownloadVerificationTest {
    private fun target(file: File, lfsOid: String?): HFDownloadSupport.DownloadTarget =
        HFDownloadSupport.DownloadTarget(
            modelFile =
                HFModelTree.HFModelFile(
                    path = "model.gguf",
                    size = file.length(),
                    lfs = lfsOid?.let { HFModelTree.HFModelFile.LfsMetadata(oid = it, size = file.length()) },
                ),
            targetFile = file,
            expectedSize = file.length(),
            expectedSha = lfsOid,
            downloadUrl = "https://example.invalid/model.gguf",
        )

    @Test
    fun `sha mismatch throws and deletes the file`() {
        val file = File.createTempFile("hf-verify", ".gguf")
        file.writeBytes(byteArrayOf(1, 2, 3, 4))

        try {
            HFDownloadSupport.verifyDownloadedFile(
                target(file, lfsOid = "0".repeat(64)),
            )
            fail("Expected sha mismatch to throw")
        } catch (expected: IllegalStateException) {
            assertTrue(expected.message.orEmpty().contains("sha mismatch"))
        }
        assertFalse("Corrupt download should be deleted", file.exists())
    }

    @Test
    fun `matching sha passes`() {
        val file = File.createTempFile("hf-verify", ".gguf")
        file.deleteOnExit()
        file.writeBytes(byteArrayOf(1, 2, 3, 4))
        // SHA-256 of bytes 01 02 03 04
        val sha = java.security.MessageDigest.getInstance("SHA-256")
            .digest(byteArrayOf(1, 2, 3, 4))
            .joinToString("") { "%02x".format(it) }

        HFDownloadSupport.verifyDownloadedFile(target(file, lfsOid = sha))

        assertTrue(file.exists())
    }

    @Test
    fun `cleanupOrphanedTempFiles removes stale temp files and keeps active and non-temp files`() {
        val dir = File.createTempFile("hf-downloads", "").let {
            it.delete(); it.mkdirs(); it
        }
        val cutoff = 1_000_000L
        val orphan = File(dir, "clip_g.safetensors-123.tmp").apply {
            writeBytes(byteArrayOf(1)); setLastModified(cutoff - 10_000L)
        }
        val active = File(dir, "t5xxl.safetensors-456.tmp").apply {
            writeBytes(byteArrayOf(1)); setLastModified(cutoff + 10_000L)
        }
        val finalized = File(dir, "model.safetensors").apply {
            writeBytes(byteArrayOf(1)); setLastModified(cutoff - 10_000L)
        }

        HFDownloadSupport.cleanupOrphanedTempFiles(dir, olderThanMillis = cutoff)

        assertFalse("Stale .tmp from a prior process should be deleted", orphan.exists())
        assertTrue("A .tmp being written by the current process should be kept", active.exists())
        assertTrue("Non-.tmp (finalized) files must never be deleted", finalized.exists())
        dir.deleteRecursively()
    }
}
