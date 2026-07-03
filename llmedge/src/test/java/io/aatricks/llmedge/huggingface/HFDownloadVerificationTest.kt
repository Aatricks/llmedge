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
}
