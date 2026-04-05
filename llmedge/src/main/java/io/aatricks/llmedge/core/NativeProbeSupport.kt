package io.aatricks.llmedge.core

internal object NativeProbeSupport {
    inline fun <T> withLoadedOrDefault(
        defaultValue: T,
        ensureLoaded: () -> Unit,
        probe: () -> T,
    ): T =
        try {
            ensureLoaded()
            probe()
        } catch (_: Throwable) {
            defaultValue
        }

    inline fun <T> unsatisfiedLinkOrDefault(
        defaultValue: T,
        probe: () -> T,
    ): T =
        try {
            probe()
        } catch (_: UnsatisfiedLinkError) {
            defaultValue
        }
}
