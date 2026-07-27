package io.aatricks.llmedge.model

import io.aatricks.llmedge.runtime.GGUFReader
import java.io.File
import kotlinx.coroutines.runBlocking

/**
 * A component family a GGUF checkpoint carries, derived from its tensor names.
 *
 * Diffusion checkpoints ship either as a bare denoiser ([DIFFUSION] alone) or as an all-in-one
 * bundle that also bakes in the text encoders and the VAE. The two are not interchangeable: a
 * bundle handed to a diffusion-model-only slot leaves the runtime loading a second, conflicting
 * copy of the encoders alongside the ones baked into the file.
 */
enum class GgufComponent {
    DIFFUSION,
    TEXT_ENCODER,
    VAE,
}

/**
 * What [GgufFileSummary.read] could recover from a GGUF file's header.
 *
 * Reads metadata and tensor names only, so the cost is independent of the weight payload. Every
 * failure degrades to `null` rather than throwing: this exists to produce a better error message,
 * and must never be the reason a valid model is rejected.
 */
data class GgufFileSummary(
    val tensorPrefixes: Set<String>,
    /** The `general.architecture` metadata value, when present. */
    val architecture: String?,
    val components: Set<GgufComponent>,
) {
    /** True when the file bundles text encoders or a VAE next to the denoiser. */
    val isAllInOne: Boolean
        get() = GgufComponent.TEXT_ENCODER in components || GgufComponent.VAE in components

    companion object {
        // Matched against GGUFReader.getTensorNamePrefixes(), which is exactly one segment deep —
        // so every key here must be a whole first segment, never a dotted path.
        private val vaePrefixes = setOf("first_stage_model", "vae")
        private val textEncoderPrefixes =
            setOf("text_encoders", "cond_stage_model", "conditioner", "clip_l", "clip_g", "t5xxl")
        private val diffusionPrefixes =
            setOf("model", "diffusion_model", "joint_blocks", "double_blocks", "single_blocks")

        @JvmStatic
        fun read(file: File): GgufFileSummary? =
            runCatching {
                GGUFReader().use { reader ->
                    runBlocking { reader.load(file.absolutePath) }
                    val prefixes = reader.getTensorNamePrefixes()
                    if (prefixes.isEmpty()) return null
                    GgufFileSummary(
                        tensorPrefixes = prefixes,
                        architecture = reader.getArchitecture(),
                        components = classify(prefixes),
                    )
                }
            }.getOrNull()

        internal fun classify(prefixes: Set<String>): Set<GgufComponent> =
            buildSet {
                if (prefixes.any(vaePrefixes::contains)) add(GgufComponent.VAE)
                if (prefixes.any(textEncoderPrefixes::contains)) add(GgufComponent.TEXT_ENCODER)
                if (prefixes.any(diffusionPrefixes::contains)) add(GgufComponent.DIFFUSION)
            }
    }
}
