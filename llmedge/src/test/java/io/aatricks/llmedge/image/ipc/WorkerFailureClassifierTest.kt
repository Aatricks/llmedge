package io.aatricks.llmedge.image.ipc

import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.core.GenerationHangException
import io.aatricks.llmedge.core.WorkerCrashedException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class WorkerFailureClassifierTest {
    @Test
    fun `watchdog kill classifies as hang with phase and backend`() {
        val result =
            WorkerFailureClassifier.classify(
                context = ApplicationProvider.getApplicationContext(),
                pid = 12345,
                lastPhase = DiffusionPhases.LOADING,
                lastBackend = "VULKAN",
                killedByWatchdog = true,
                stallMs = 90_000L,
            )
        assertTrue(result is GenerationHangException)
        result as GenerationHangException
        assertEquals("VULKAN", result.backend)
        assertEquals(DiffusionPhases.LOADING, result.phase)
        assertEquals(90_000L, result.stallMs)
    }

    @Test
    fun `unknown exit reason classifies as crash`() {
        val result =
            WorkerFailureClassifier.classify(
                context = ApplicationProvider.getApplicationContext(),
                pid = 12345,
                lastPhase = DiffusionPhases.GENERATING,
                lastBackend = "CPU",
                killedByWatchdog = false,
                stallMs = 0L,
            )
        assertTrue(result is WorkerCrashedException)
        assertEquals("CPU", (result as WorkerCrashedException).backend)
    }

    /**
     * Reproduces llmedge-examples#37: a REASON_CRASH worker death that left no tombstone (not a
     * native crash) and no breadcrumb (the uncaught handler never ran), which previously reduced
     * the whole post-mortem to the word "crash".
     */
    @Test
    fun `crash summary carries the worker log trail when no breadcrumb exists`() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val pid = 4242
        WorkerDiagnosticsLog.logFile(context, pid).writeText(
            "[100] I/DiffusionWorker: Worker engine initialized\n" +
                "[200] I/DiffusionRuntimeLoader: Creating managed CPU runtime for t5xxl sequential=true\n",
        )

        val result =
            WorkerFailureClassifier.classify(
                context = context,
                pid = pid,
                lastPhase = DiffusionPhases.GENERATING,
                lastBackend = "CPU",
                killedByWatchdog = false,
                stallMs = 0L,
            ) as WorkerCrashedException

        val summary = requireNotNull(result.crashSummary)
        assertTrue(summary, summary.contains("worker-log:"))
        assertTrue(summary, summary.contains("Creating managed CPU runtime for t5xxl"))
        assertTrue("trail should be consumed", !WorkerDiagnosticsLog.logFile(context, pid).exists())
    }

    @Test
    fun `crash summary omits the trail section when the worker left no log`() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()

        val result =
            WorkerFailureClassifier.classify(
                context = context,
                pid = 5150,
                lastPhase = DiffusionPhases.GENERATING,
                lastBackend = "CPU",
                killedByWatchdog = false,
                stallMs = 0L,
            ) as WorkerCrashedException

        assertTrue(requireNotNull(result.crashSummary).let { !it.contains("worker-log:") })
    }
}
