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
}
