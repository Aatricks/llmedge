package io.aatricks.llmedge.image.ipc

import io.aatricks.llmedge.WorkerWatchdogConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GenerationWatchdogTest {
    private class Harness(
        config: WorkerWatchdogConfig = WorkerWatchdogConfig(),
    ) {
        var nowMs = 0L
        var cpuMs: Long? = 0L
        var verdicts = mutableListOf<Triple<String, String?, Long>>()
        val watchdog =
            GenerationWatchdog(
                config = config,
                clock = { nowMs },
                cpuTimeMsReader = { cpuMs },
            ) { phase, backend, stallMs -> verdicts.add(Triple(phase, backend, stallMs)) }

        /** Advance time in sample-interval ticks, optionally accruing CPU. */
        fun run(
            durationMs: Long,
            cpuBusyFraction: Double,
            stepMs: Long = 2_000,
        ) {
            var elapsed = 0L
            while (elapsed < durationMs) {
                nowMs += stepMs
                cpuMs = cpuMs?.plus((stepMs * cpuBusyFraction).toLong())
                elapsed += stepMs
                watchdog.sample()
            }
        }
    }

    @Test
    fun `flat cpu loading hang fires at the loading stall timeout`() {
        val h = Harness()
        h.watchdog.onPhase(DiffusionPhases.LOADING, "VULKAN")
        h.run(durationMs = 95_000, cpuBusyFraction = 0.0)
        assertEquals(1, h.verdicts.size)
        val (phase, backend, _) = h.verdicts.single()
        assertEquals(DiffusionPhases.LOADING, phase)
        assertEquals("VULKAN", backend)
    }

    @Test
    fun `busy cpu cold shader compile never fires`() {
        val h = Harness()
        h.watchdog.onPhase(DiffusionPhases.LOADING, "VULKAN")
        // Pixel 8 signature: ~310s before the first step, one core pegged.
        h.run(durationMs = 310_000, cpuBusyFraction = 1.0)
        assertTrue(h.verdicts.isEmpty())
    }

    @Test
    fun `step heartbeats keep resetting the window`() {
        val h = Harness()
        h.watchdog.onPhase(DiffusionPhases.GENERATING, "VULKAN")
        repeat(10) {
            h.run(durationMs = 60_000, cpuBusyFraction = 0.0)
            h.watchdog.onStep(it, 10)
        }
        assertTrue(h.verdicts.isEmpty())
        // Steps stop AND cpu is flat -> verdict after the step stall window.
        h.run(durationMs = 310_000, cpuBusyFraction = 0.0)
        assertEquals(1, h.verdicts.size)
        assertEquals(DiffusionPhases.STEP, h.verdicts.single().first)
    }

    @Test
    fun `resolving phase is exempt from the flat cpu rule`() {
        val h = Harness()
        h.watchdog.onPhase(DiffusionPhases.RESOLVING_MODEL, null)
        // A model download: CPU-flat for 20 minutes. Must not fire (hard wall is 30 min).
        h.run(durationMs = 20 * 60_000, cpuBusyFraction = 0.0)
        assertTrue(h.verdicts.isEmpty())
    }

    @Test
    fun `hard wall fires regardless of cpu activity`() {
        val h = Harness()
        h.watchdog.onPhase(DiffusionPhases.RESOLVING_MODEL, null)
        h.run(durationMs = 31 * 60_000, cpuBusyFraction = 1.0)
        assertEquals(1, h.verdicts.size)
    }

    @Test
    fun `unreadable cpu never fires the stall rule`() {
        val h = Harness()
        h.cpuMs = null
        h.watchdog.onPhase(DiffusionPhases.LOADING, "VULKAN")
        h.run(durationMs = 10 * 60_000, cpuBusyFraction = 0.0)
        assertTrue(h.verdicts.isEmpty())
    }

    @Test
    fun `verdict fires exactly once and stop disarms`() {
        val h = Harness()
        h.watchdog.onPhase(DiffusionPhases.LOADING, "VULKAN")
        h.run(durationMs = 200_000, cpuBusyFraction = 0.0)
        assertEquals(1, h.verdicts.size)

        val h2 = Harness()
        h2.watchdog.onPhase(DiffusionPhases.LOADING, "VULKAN")
        h2.watchdog.stop()
        h2.run(durationMs = 200_000, cpuBusyFraction = 0.0)
        assertTrue(h2.verdicts.isEmpty())
    }

    @Test
    fun `low but nonzero cpu below threshold still counts as hung`() {
        val h = Harness()
        h.watchdog.onPhase(DiffusionPhases.GENERATING, "VULKAN")
        h.run(durationMs = 100_000, cpuBusyFraction = 0.005)
        assertFalse(h.verdicts.isEmpty())
    }
}
