package io.aatricks.llmedge.image

/**
 * Coarse lifecycle heartbeats for a diffusion request, consumed by the worker-process watchdog.
 * Phases are the [io.aatricks.llmedge.image.ipc.DiffusionPhases] constants. Implementations must
 * be cheap and non-blocking: callbacks fire on the generation path (steps come from the native
 * per-denoise-step progress callback).
 */
internal interface DiffusionPhaseListener {
    fun onPhase(
        phase: String,
        backend: String? = null,
    )

    fun onStep(
        step: Int,
        totalSteps: Int,
    )
}
