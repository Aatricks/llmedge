package io.aatricks.llmedge.huggingface

import android.app.DownloadManager
import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device end-to-end proof that an interrupted system download resumes instead of restarting.
 *
 * Downloads a real multi-GB file, so it only runs when explicitly requested:
 * `-e llmedge.resumeE2E 1`. The two phases are separate `am instrument` invocations — the
 * instrumentation process dies in between, which IS the interruption under test:
 *
 *  1. [phase1_startAndDetachMidFlight] starts the download, waits until it is genuinely
 *     mid-flight, cancels the caller (must detach, not abort), and records {downloadId, bytes}.
 *  2. [phase2_reconnectAndComplete] (run later, fresh process) asserts the same DownloadManager
 *     record survived and progressed while no app code was running, then re-invokes the exact
 *     production API and asserts it reconnects (same id, no restart) and finalizes through the
 *     byte/SHA verification gate.
 */
@RunWith(AndroidJUnit4::class)
class SystemDownloadResumeE2ETest {
    private lateinit var context: Context
    private lateinit var stateFile: File

    private val modelId = "unsloth/Qwen3-4B-GGUF"
    private val wantedSuffix = "Q4_K_M.gguf"

    @Before
    fun setUp() {
        assumeTrue(
            "resume E2E only runs with -e llmedge.resumeE2E 1",
            InstrumentationRegistry.getArguments().getString("llmedge.resumeE2E") == "1",
        )
        context = ApplicationProvider.getApplicationContext()
        stateFile = File(context.filesDir, "resume-e2e-state.txt")
    }

    private fun ourDownloads(): List<Triple<Long, Int, Long>> {
        val dm = context.getSystemService(DownloadManager::class.java)!!
        val out = mutableListOf<Triple<Long, Int, Long>>()
        dm.query(DownloadManager.Query()).use { c ->
            val id = c.getColumnIndex(DownloadManager.COLUMN_ID)
            val status = c.getColumnIndex(DownloadManager.COLUMN_STATUS)
            val soFar = c.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
            while (c.moveToNext()) {
                out.add(Triple(c.getLong(id), c.getInt(status), c.getLong(soFar)))
            }
        }
        return out
    }

    private suspend fun ensure() =
        HFDownloadSupport.ensureFileOnDisk(
            destinationRoot = File(context.filesDir, "hf-models"),
            modelId = modelId,
            revision = "main",
            token = null,
            forceDownload = false,
            preferSystemDownloader = true,
            systemDownloadContext = context,
            onProgress = null,
            noMatchMessage = "no $wantedSuffix in $modelId",
        ) { files -> files.firstOrNull { it.path.endsWith(wantedSuffix) } }

    @Test
    fun phase1_startAndDetachMidFlight() {
        val before = ourDownloads().map { it.first }.toSet()
        val scope = CoroutineScope(Dispatchers.IO + Job())
        val job = scope.launch { runCatching { ensure() } }

        // Wait until the download is genuinely mid-flight (registered + >64 MB in).
        var mine: Triple<Long, Int, Long>? = null
        val start = SystemClock.uptimeMillis()
        while (SystemClock.uptimeMillis() - start < 180_000) {
            mine = ourDownloads().firstOrNull { it.first !in before && it.third > 64L * 1024 * 1024 }
            if (mine != null) break
            Thread.sleep(1_000)
        }
        assertTrue("download never reached mid-flight (64MB)", mine != null)

        // Cancel the caller: the monitor must detach, leaving the system download alive.
        runBlocking { job.cancelAndJoin() }
        Thread.sleep(3_000)
        val after = ourDownloads().firstOrNull { it.first == mine!!.first }
        assertTrue("system download must survive caller cancellation", after != null)
        assertTrue(
            "download must still be live after cancel, status=${after!!.second}",
            after.second != DownloadManager.STATUS_FAILED,
        )

        stateFile.writeText("${after.first} ${after.third} ${System.currentTimeMillis()}")
        Log.i("ResumeE2E", "phase1 done: id=${after.first} bytes=${after.third}")
    }

    @Test
    fun phase2_reconnectAndComplete() {
        assumeTrue("phase1 state missing — run phase1 first", stateFile.isFile)
        val (idStr, bytesStr, _) = stateFile.readText().trim().split(" ")
        val phase1Id = idStr.toLong()
        val phase1Bytes = bytesStr.toLong()

        val record = ourDownloads().firstOrNull { it.first == phase1Id }
        assertTrue("phase1 download record must survive process death", record != null)
        assertTrue(
            "bytes must have progressed while no app code ran " +
                "(phase1=$phase1Bytes, now=${record!!.third}, status=${record.second})",
            record.third > phase1Bytes || record.second == DownloadManager.STATUS_SUCCESSFUL,
        )
        Log.i("ResumeE2E", "phase2 start: id=$phase1Id grew ${phase1Bytes} -> ${record.third}")

        val result = runBlocking { withTimeout(20 * 60_000L) { ensure() } }
        val file = result.file
        assertTrue("finalized file must exist", file.isFile)
        assertTrue("finalized file must be complete, got ${file.length()}", file.length() > 2_000_000_000L)

        // Reconnect proof: no second DownloadManager record was created for this file.
        val newIds = ourDownloads().filter { it.first > phase1Id }
        assertEquals(
            "no new download may be enqueued during resume, got $newIds",
            0,
            newIds.size,
        )
        stateFile.delete()
        Log.i("ResumeE2E", "phase2 done: finalized ${file.name} (${file.length()} bytes)")
    }
}
