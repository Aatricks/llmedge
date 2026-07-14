package io.aatricks.llmedge.huggingface

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Test

class SystemDownloadTest {
    @Test
    fun `recognizes a complete payload before DownloadManager marks success`() {
        val file = File.createTempFile("system-download", ".tmp")
        file.deleteOnExit()
        file.writeBytes(ByteArray(4))

        assertTrue(
            SystemDownload.hasCompletePayload(
                destination = file,
                downloadedBytes = 4,
                totalBytes = 4,
            ),
        )
    }
}
