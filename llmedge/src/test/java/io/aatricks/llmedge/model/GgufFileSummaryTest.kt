package io.aatricks.llmedge.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.File

class GgufFileSummaryTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private var fileCounter = 0

    @Test
    fun `reads architecture and classifies a diffusion-only checkpoint`() {
        val file = writeGguf(
            architecture = "sd3",
            tensorNames = listOf("joint_blocks.0.x_block.attn.qkv.weight", "model.diffusion_model.pos_embed"),
        )

        val summary = GgufFileSummary.read(file)!!

        assertEquals("sd3", summary.architecture)
        assertEquals(2L, summary.tensorCount)
        assertEquals(setOf(GgufComponent.DIFFUSION), summary.components)
        assertFalse(summary.isAllInOne)
    }

    @Test
    fun `flags a bundle carrying text encoders and a VAE`() {
        val file = writeGguf(
            architecture = "sd3",
            tensorNames = listOf(
                "model.diffusion_model.joint_blocks.0.weight",
                "text_encoders.clip_l.transformer.weight",
                "first_stage_model.decoder.conv_in.weight",
            ),
        )

        val summary = GgufFileSummary.read(file)!!

        assertEquals(
            setOf(GgufComponent.DIFFUSION, GgufComponent.TEXT_ENCODER, GgufComponent.VAE),
            summary.components,
        )
        assertTrue(summary.isAllInOne)
    }

    @Test
    fun `skips metadata value types it does not read`() {
        val file = writeGguf(
            architecture = "flux",
            tensorNames = listOf("model.diffusion_model.double_blocks.0.weight"),
            extraMetadata = listOf(
                "some.uint32" to KvValue.UInt32(7),
                "some.float32" to KvValue.Float32(1.5f),
                "some.bool" to KvValue.Bool(true),
                "some.uint64" to KvValue.UInt64(1L shl 40),
                "some.strings" to KvValue.StringArray(listOf("alpha", "beta")),
                "some.ints" to KvValue.UInt32Array(listOf(1, 2, 3)),
            ),
        )

        val summary = GgufFileSummary.read(file)!!

        assertEquals("flux", summary.architecture)
        assertEquals(setOf(GgufComponent.DIFFUSION), summary.components)
    }

    @Test
    fun `returns null for a non-gguf file`() {
        val file = temporaryFolder.newFile("not-a-model.gguf").apply { writeText("definitely not gguf") }

        assertNull(GgufFileSummary.read(file))
    }

    @Test
    fun `returns null for a truncated header rather than throwing`() {
        val complete = writeGguf(architecture = "sd3", tensorNames = listOf("model.diffusion_model.a.weight"))
        val truncated = temporaryFolder.newFile("truncated.gguf")
        truncated.writeBytes(complete.readBytes().copyOf(20))

        assertNull(GgufFileSummary.read(truncated))
    }

    @Test
    fun `reports no components for an unrecognised tensor layout`() {
        val file = writeGguf(architecture = "unknown", tensorNames = listOf("mystery.weight"))

        val summary = GgufFileSummary.read(file)!!

        assertTrue(summary.components.isEmpty())
        assertFalse(summary.isAllInOne)
    }

    private sealed interface KvValue {
        data class Str(val value: String) : KvValue

        data class UInt32(val value: Int) : KvValue

        data class Float32(val value: Float) : KvValue

        data class Bool(val value: Boolean) : KvValue

        data class UInt64(val value: Long) : KvValue

        data class StringArray(val values: List<String>) : KvValue

        data class UInt32Array(val values: List<Int>) : KvValue
    }

    /** Builds a minimal but spec-shaped GGUF v3 header; the weight payload is irrelevant here. */
    private fun writeGguf(
        architecture: String?,
        tensorNames: List<String>,
        extraMetadata: List<Pair<String, KvValue>> = emptyList(),
    ): File {
        val out = ByteArrayOutputStream()
        out.write(byteArrayOf('G'.code.toByte(), 'G'.code.toByte(), 'U'.code.toByte(), 'F'.code.toByte()))
        out.writeUInt32(3)
        out.writeUInt64(tensorNames.size.toLong())

        val metadata = buildList {
            architecture?.let { add("general.architecture" to KvValue.Str(it)) }
            addAll(extraMetadata)
        }
        out.writeUInt64(metadata.size.toLong())
        metadata.forEach { (key, value) ->
            out.writeGgufString(key)
            out.writeKv(value)
        }

        tensorNames.forEach { name ->
            out.writeGgufString(name)
            out.writeUInt32(2) // dimension count
            out.writeUInt64(16)
            out.writeUInt64(16)
            out.writeUInt32(0) // ggml type F32
            out.writeUInt64(0) // offset
        }
        out.write(ByteArray(32)) // stand-in for the weight payload

        return temporaryFolder.newFile("model-${fileCounter++}.gguf").apply {
            writeBytes(out.toByteArray())
        }
    }

    private fun ByteArrayOutputStream.writeKv(value: KvValue) {
        when (value) {
            is KvValue.Str -> {
                writeUInt32(8)
                writeGgufString(value.value)
            }
            is KvValue.UInt32 -> {
                writeUInt32(4)
                writeUInt32(value.value.toLong())
            }
            is KvValue.Float32 -> {
                writeUInt32(6)
                writeUInt32(java.lang.Float.floatToIntBits(value.value).toLong())
            }
            is KvValue.Bool -> {
                writeUInt32(7)
                write(if (value.value) 1 else 0)
            }
            is KvValue.UInt64 -> {
                writeUInt32(10)
                writeUInt64(value.value)
            }
            is KvValue.StringArray -> {
                writeUInt32(9)
                writeUInt32(8)
                writeUInt64(value.values.size.toLong())
                value.values.forEach { writeGgufString(it) }
            }
            is KvValue.UInt32Array -> {
                writeUInt32(9)
                writeUInt32(4)
                writeUInt64(value.values.size.toLong())
                value.values.forEach { writeUInt32(it.toLong()) }
            }
        }
    }

    private fun ByteArrayOutputStream.writeGgufString(value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        writeUInt64(bytes.size.toLong())
        write(bytes)
    }

    private fun ByteArrayOutputStream.writeUInt32(value: Long) {
        for (index in 0 until 4) {
            write(((value shr (index * 8)) and 0xFF).toInt())
        }
    }

    private fun ByteArrayOutputStream.writeUInt64(value: Long) {
        for (index in 0 until 8) {
            write(((value shr (index * 8)) and 0xFF).toInt())
        }
    }
}
