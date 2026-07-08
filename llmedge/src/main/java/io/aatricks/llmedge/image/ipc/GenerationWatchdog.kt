package io.aatricks.llmedge.image.ipc

import io.aatricks.llmedge.WorkerWatchdogConfig
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Pure hang detector for one generation request. Driven from outside: phase/step heartbeats via
 * [onPhase]/[onStep], periodic [sample] ticks (every [WorkerWatchdogConfig.cpuSampleIntervalMs]).
 *
 * Hang verdict = no heartbeat for the current phase's stall window AND the worker's CPU delta over
 * that window is flat. Flat CPU is the discriminator: a cold shader compile pegs a core for
 * minutes and must never be killed; a driver dispatch deadlock sits at 0% with all threads
 * sleeping. RESOLVING_MODEL (network download) is legitimately CPU-flat and only the hard wall
 * applies there. If CPU time is unreadable the stall rule is skipped entirely (hard wall only).
 */
internal class GenerationWatchdog(
    private val config: WorkerWatchdogConfig,
    private val clock: () -> Long,
    private val cpuTimeMsReader: () -> Long?,
    private val onHangVerdict: (phase: String, backend: String?, stallMs: Long) -> Unit,
) {
    private val fired = AtomicBoolean(false)
    private val stopped = AtomicBoolean(false)

    private val startMs = clock()

    @Volatile private var phase: String = DiffusionPhases.REQUESTED

    @Volatile private var backend: String? = null

    @Volatile private var windowStartMs: Long = startMs

    @Volatile private var cpuAtWindowStartMs: Long? = null

    @Synchronized
    fun onPhase(
        newPhase: String,
        newBackend: String?,
    ) {
        phase = newPhase
        if (newBackend != null) backend = newBackend
        resetWindow()
    }

    fun onStep(
        step: Int,
        totalSteps: Int,
    ) = onPhase(DiffusionPhases.STEP, null)

    @Synchronized
    fun sample() {
        if (fired.get() || stopped.get()) return
        val now = clock()
        if (now - startMs >= config.hardWallTimeoutMs) {
            fire(now)
            return
        }
        val stallMs = now - windowStartMs
        if (stallMs < stallTimeoutFor(phase)) return
        if (phase == DiffusionPhases.RESOLVING_MODEL) return // downloads are CPU-flat; hard wall only

        val cpuNow = cpuTimeMsReader() ?: return // unreadable: never kill on the stall rule
        val cpuAtStart = cpuAtWindowStartMs
        if (cpuAtStart == null) {
            // First readable sample inside this window: anchor and re-measure from here.
            cpuAtWindowStartMs = cpuNow
            windowStartMs = now
            return
        }
        val busyFraction = (cpuNow - cpuAtStart).toDouble() / stallMs.toDouble()
        if (busyFraction < config.flatCpuThreshold) {
            fire(now)
        } else {
            // Worker is computing (e.g. shader compile). Slide the window so flatness is always
            // judged over the most recent stall period.
            cpuAtWindowStartMs = cpuNow
            windowStartMs = now
        }
    }

    fun stop() {
        stopped.set(true)
    }

    fun lastPhase(): String = phase

    fun lastBackend(): String? = backend

    private fun fire(now: Long) {
        if (fired.compareAndSet(false, true)) {
            onHangVerdict(phase, backend, now - windowStartMs)
        }
    }

    private fun resetWindow() {
        windowStartMs = clock()
        cpuAtWindowStartMs = cpuTimeMsReader()
    }

    private fun stallTimeoutFor(phase: String): Long =
        when (phase) {
            DiffusionPhases.RESOLVING_MODEL -> config.resolvingStallTimeoutMs
            DiffusionPhases.LOADING -> config.loadingStallTimeoutMs
            DiffusionPhases.GENERATING -> config.generatingStallTimeoutMs
            DiffusionPhases.STEP -> config.stepStallTimeoutMs
            else -> config.loadingStallTimeoutMs
        }
}
