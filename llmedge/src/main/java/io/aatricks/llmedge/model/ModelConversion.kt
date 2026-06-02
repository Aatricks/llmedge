package io.aatricks.llmedge.model

/**
 * Target precision for an import-time safetensors → GGUF conversion.
 *
 * `F16` is "direct" (no precision loss); the others are progressively smaller "lossy" quantizations.
 * Labels match the `--precision` values of `tools/safetensors-convert/convert.py`.
 */
enum class ConversionPrecision(val ggufLabel: String) {
    F16("f16"),
    Q8_0("q8_0"),
    Q4_K_M("q4_k_m"),
    IQ2_BN("iq2_bn"),
    IQ2_BN_R4("iq2_bn_r4"),
}

/** Model-specific pre-processing applied before the stock safetensors → GGUF conversion. */
enum class ConversionAdapter(val cliFlag: String?) {
    /** Stock Hugging Face architecture; no special handling. */
    NONE(null),

    /** Bonsai / QLlama: fold per-output `.scales` into the weights before conversion. */
    BONSAI_QLINEAR("bonsai-qlinear"),
}

/**
 * Declares that a [ModelSpec] points at a safetensors source that must be converted to GGUF before the
 * runtime can load it. Attached via [ModelHints.conversion] by [ModelSpec.safetensors].
 */
data class ModelConversion(
    val precision: ConversionPrecision = ConversionPrecision.F16,
    val adapter: ConversionAdapter = ConversionAdapter.NONE,
) {
    /** Stable token distinguishing one conversion target from another in cache keys. */
    val cacheToken: String
        get() = "convert:${precision.ggufLabel}:${adapter.name.lowercase()}"
}
