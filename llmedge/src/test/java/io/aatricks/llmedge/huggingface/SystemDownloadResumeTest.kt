package io.aatricks.llmedge.huggingface

import android.app.DownloadManager
import io.aatricks.llmedge.huggingface.SystemDownload.ResumeAction
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SystemDownloadResumeTest {
    private fun existing(
        status: Int,
        totalBytes: Long = 100L,
        localPath: String? = "/data/hf/model.tmp",
    ) = SystemDownload.ExistingDownload(
        id = 42L,
        status = status,
        bytesDownloaded = totalBytes,
        totalBytes = totalBytes,
        localPath = localPath,
    )

    @Test
    fun `finished download with complete file is finalized`() {
        assertEquals(
            ResumeAction.FINALIZE,
            SystemDownload.decideResume(existing(DownloadManager.STATUS_SUCCESSFUL, totalBytes = 100L), localLength = 100L),
        )
    }

    @Test
    fun `finished download with truncated file is re-enqueued`() {
        assertEquals(
            ResumeAction.REENQUEUE,
            SystemDownload.decideResume(existing(DownloadManager.STATUS_SUCCESSFUL, totalBytes = 100L), localLength = 50L),
        )
    }

    @Test
    fun `finished download with unknown size is re-enqueued`() {
        assertEquals(
            ResumeAction.REENQUEUE,
            SystemDownload.decideResume(existing(DownloadManager.STATUS_SUCCESSFUL, totalBytes = 0L), localLength = 100L),
        )
    }

    @Test
    fun `finished download with missing local path is re-enqueued`() {
        assertEquals(
            ResumeAction.REENQUEUE,
            SystemDownload.decideResume(existing(DownloadManager.STATUS_SUCCESSFUL, localPath = null), localLength = 100L),
        )
    }

    @Test
    fun `in-flight downloads are monitored`() {
        for (status in intArrayOf(
            DownloadManager.STATUS_RUNNING,
            DownloadManager.STATUS_PENDING,
            DownloadManager.STATUS_PAUSED,
        )) {
            assertEquals(
                "status=$status should reconnect",
                ResumeAction.MONITOR,
                SystemDownload.decideResume(existing(status), localLength = 40L),
            )
        }
    }

    @Test
    fun `in-flight download without a local path is re-enqueued`() {
        assertEquals(
            ResumeAction.REENQUEUE,
            SystemDownload.decideResume(existing(DownloadManager.STATUS_RUNNING, localPath = null), localLength = 0L),
        )
    }

    @Test
    fun `failed download is re-enqueued`() {
        assertEquals(
            ResumeAction.REENQUEUE,
            SystemDownload.decideResume(existing(DownloadManager.STATUS_FAILED), localLength = 40L),
        )
    }

    @Test
    fun `a resumable match beats a failed one regardless of order`() {
        val failed = existing(DownloadManager.STATUS_FAILED)
        val running = existing(DownloadManager.STATUS_RUNNING)
        assertTrue(SystemDownload.isBetterMatch(running, failed))
        assertFalse(SystemDownload.isBetterMatch(failed, running))
    }

    @Test
    fun `among resumable matches the one further along wins`() {
        val behind = existing(DownloadManager.STATUS_RUNNING).copy(bytesDownloaded = 10L)
        val ahead = existing(DownloadManager.STATUS_RUNNING).copy(bytesDownloaded = 80L)
        assertTrue(SystemDownload.isBetterMatch(ahead, behind))
        assertFalse(SystemDownload.isBetterMatch(behind, ahead))
    }
}
