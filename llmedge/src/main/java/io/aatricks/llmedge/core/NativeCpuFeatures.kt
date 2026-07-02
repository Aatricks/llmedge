package io.aatricks.llmedge.core

/**
 * CPU feature probe backed by `getauxval(AT_HWCAP/AT_HWCAP2)` via the GGUF reader
 * library (which is always compiled for the baseline ISA and therefore safe to load
 * before any feature detection has happened).
 *
 * Kernel hwcaps are the authoritative source for ISA features; `/proc/cpuinfo`
 * string-scraping remains only as a fallback when the probe library is unavailable.
 */
internal object NativeCpuFeatures {
    // Linux arm64 hwcap bits (uapi/asm-arm64/hwcap.h).
    private const val HWCAP_ASIMD = 1L shl 1
    private const val HWCAP_AES = 1L shl 3
    private const val HWCAP_CRC32 = 1L shl 7
    private const val HWCAP_FPHP = 1L shl 9
    private const val HWCAP_ASIMDHP = 1L shl 10
    private const val HWCAP_DCPOP = 1L shl 16
    private const val HWCAP_ASIMDDP = 1L shl 20
    private const val HWCAP_USCAT = 1L shl 25
    private const val HWCAP2_I8MM = 1L shl 13

    data class Features(
        val hasFp16: Boolean,
        val hasDotProd: Boolean,
        val hasI8mm: Boolean,
        val isAtLeastArmV82: Boolean,
        val isAtLeastArmV84: Boolean,
    )

    /**
     * Returns kernel-reported features, or null when the probe is unavailable
     * (library failed to load, non-Linux host, or hwcaps empty).
     */
    val features: Features? by lazy { readFeaturesOrNull() }

    private fun readFeaturesOrNull(): Features? {
        val caps =
            try {
                NativeLibraryLoader.ensureGgufReaderLoaded(
                    required = false,
                    onDebug = {},
                    onError = { _, _ -> },
                )
                nativeGetHwcaps()
            } catch (_: Throwable) {
                // UnsatisfiedLinkError when libggufreader is absent, or any loader issue.
                null
            } ?: return null
        if (caps.size < 2) return null
        val hwcap = caps[0]
        val hwcap2 = caps[1]
        if (hwcap == 0L) return null
        return Features(
            hasFp16 = hwcap and (HWCAP_FPHP or HWCAP_ASIMDHP) != 0L,
            hasDotProd = hwcap and HWCAP_ASIMDDP != 0L,
            hasI8mm = hwcap2 and HWCAP2_I8MM != 0L,
            isAtLeastArmV82 =
                hwcap and HWCAP_ASIMD != 0L &&
                    hwcap and HWCAP_CRC32 != 0L &&
                    hwcap and HWCAP_AES != 0L,
            isAtLeastArmV84 = hwcap and HWCAP_DCPOP != 0L && hwcap and HWCAP_USCAT != 0L,
        )
    }

    @JvmStatic
    private external fun nativeGetHwcaps(): LongArray?
}
