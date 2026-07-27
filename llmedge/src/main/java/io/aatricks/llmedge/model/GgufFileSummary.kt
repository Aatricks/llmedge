package io.aatricks.llmedge.model

import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.InputStream

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
 * Only the metadata and tensor-info sections are read, so the cost is independent of the weight
 * payload. Every failure degrades to `null` rather than throwing: this exists to produce a better
 * error message, and must never be the reason a valid model is rejected.
 *
 * Deliberately not built on [io.aatricks.llmedge.runtime.GGUFReader], which overlaps only on
 * `general.architecture`: that reader enumerates no tensors, so it cannot tell a bare denoiser
 * from a bundle, and it needs its native library — which would put a `.so` dependency on the
 * import path and take this classification out of reach of JVM unit tests.
 */
data class GgufFileSummary(
    val version: Int,
    val tensorCount: Long,
    /** The `general.architecture` metadata value, when present. */
    val architecture: String?,
    val components: Set<GgufComponent>,
) {
    /** True when the file bundles text encoders or a VAE next to the denoiser. */
    val isAllInOne: Boolean
        get() = GgufComponent.TEXT_ENCODER in components || GgufComponent.VAE in components

    companion object {
        private const val MAGIC = 0x46554747L // "GGUF" read little-endian.
        private const val MIN_VERSION = 2

        /** Guards against a corrupt count turning into an unbounded read. */
        private const val MAX_TENSORS = 1_000_000L
        private const val MAX_KV_PAIRS = 100_000L
        private const val MAX_ARRAY_LENGTH = 50_000_000L
        private const val MAX_STRING_BYTES = 1L shl 20

        private val vaePrefixes = listOf("first_stage_model.", "vae.")
        private val textEncoderPrefixes =
            listOf("text_encoders.", "cond_stage_model.", "conditioner.", "clip_l.", "clip_g.", "t5xxl.")
        private val diffusionPrefixes = listOf("model.diffusion_model.", "diffusion_model.")

        @JvmStatic
        fun read(file: File): GgufFileSummary? =
            runCatching {
                BufferedInputStream(FileInputStream(file), 1 shl 16).use(::parse)
            }.getOrNull()

        private fun parse(input: InputStream): GgufFileSummary? {
            if (input.readUInt32() != MAGIC) return null
            val version = input.readUInt32().toInt()
            if (version < MIN_VERSION) return null

            val tensorCount = input.readUInt64()
            val kvCount = input.readUInt64()
            if (tensorCount !in 0..MAX_TENSORS || kvCount !in 0..MAX_KV_PAIRS) return null

            var architecture: String? = null
            repeat(kvCount.toInt()) {
                val key = input.readGgufString() ?: return null
                val value = input.readKvValue() ?: return null
                if (key == "general.architecture" && value is String) {
                    architecture = value
                }
            }

            val components = mutableSetOf<GgufComponent>()
            repeat(tensorCount.toInt()) {
                val name = input.readGgufString() ?: return null
                classify(name)?.let(components::add)
                val dimensions = input.readUInt32().toInt()
                if (dimensions !in 0..8) return null
                repeat(dimensions) { input.readUInt64() }
                input.readUInt32() // ggml type
                input.readUInt64() // offset
            }

            return GgufFileSummary(
                version = version,
                tensorCount = tensorCount,
                architecture = architecture,
                components = components,
            )
        }

        private fun classify(tensorName: String): GgufComponent? =
            when {
                vaePrefixes.any(tensorName::startsWith) -> GgufComponent.VAE
                textEncoderPrefixes.any(tensorName::startsWith) -> GgufComponent.TEXT_ENCODER
                diffusionPrefixes.any(tensorName::startsWith) -> GgufComponent.DIFFUSION
                else -> null
            }

        /**
         * Skips a metadata value, materialising it only for the scalar string case the caller
         * reads. Arrays recurse one level, which is all GGUF allows.
         */
        private fun InputStream.readKvValue(): Any? =
            when (readUInt32().toInt()) {
                0, 1, 7 -> readExactly(1) // uint8 / int8 / bool
                2, 3 -> readExactly(2) // uint16 / int16
                4, 5, 6 -> readExactly(4) // uint32 / int32 / float32
                8 -> readGgufString()
                9 -> readArrayValue()
                10, 11, 12 -> readExactly(8) // uint64 / int64 / float64
                else -> null
            }

        private fun InputStream.readArrayValue(): Any? {
            val elementType = readUInt32().toInt()
            val length = readUInt64()
            if (length !in 0..MAX_ARRAY_LENGTH) return null
            repeat(length.toInt()) {
                val element: Any? =
                    when (elementType) {
                        0, 1, 7 -> readExactly(1)
                        2, 3 -> readExactly(2)
                        4, 5, 6 -> readExactly(4)
                        8 -> readGgufString()
                        10, 11, 12 -> readExactly(8)
                        else -> null
                    }
                element ?: return null
            }
            return Unit
        }

        private fun InputStream.readGgufString(): String? {
            val length = readUInt64()
            if (length !in 0..MAX_STRING_BYTES) return null
            val bytes = readExactly(length.toInt()) ?: return null
            return String(bytes, Charsets.UTF_8)
        }

        private fun InputStream.readUInt32(): Long = readLittleEndian(4)

        private fun InputStream.readUInt64(): Long = readLittleEndian(8)

        private fun InputStream.readLittleEndian(byteCount: Int): Long {
            val bytes = readExactly(byteCount) ?: return -1L
            var value = 0L
            for (index in byteCount - 1 downTo 0) {
                value = (value shl 8) or (bytes[index].toLong() and 0xFF)
            }
            return value
        }

        private fun InputStream.readExactly(count: Int): ByteArray? {
            if (count < 0) return null
            val buffer = ByteArray(count)
            var read = 0
            while (read < count) {
                val n = read(buffer, read, count - read)
                if (n < 0) return null
                read += n
            }
            return buffer
        }
    }
}
