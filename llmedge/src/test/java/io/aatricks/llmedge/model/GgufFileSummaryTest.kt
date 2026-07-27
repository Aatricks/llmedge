package io.aatricks.llmedge.model

import io.aatricks.llmedge.core.InvalidModelFileException
import io.aatricks.llmedge.runtime.GGUFReader
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Classification is asserted against the tensor-name prefixes the native reader returns. The
 * DEPTH CONTRACT those keys depend on — exactly one segment — lives in
 * `gguf_reader_internal.cpp`; the fixtures below use real checkpoint layouts so a change on
 * either side shows up here.
 */
class GgufFileSummaryTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private var counter = 0

    @After
    fun tearDown() {
        GGUFReader.resetNativeBridgeForTests()
    }

    @Test
    fun `city96 style DiT-only checkpoint is not an all-in-one`() {
        stubReader("sd3", setOf("joint_blocks", "pos_embed", "x_embedder", "final_layer"))

        val summary = GgufFileSummary.read(ggufFile())!!

        assertEquals("sd3", summary.architecture)
        assertEquals(setOf(GgufComponent.DIFFUSION), summary.components)
        assertFalse(summary.isAllInOne)
    }

    @Test
    fun `second-state style bundle is flagged as all-in-one`() {
        stubReader("sd3", setOf("model", "text_encoders", "first_stage_model"))

        val summary = GgufFileSummary.read(ggufFile())!!

        assertEquals(
            setOf(GgufComponent.DIFFUSION, GgufComponent.TEXT_ENCODER, GgufComponent.VAE),
            summary.components,
        )
        assertTrue(summary.isAllInOne)
    }

    @Test
    fun `a text-encoder-only checkpoint claims no diffusion components`() {
        stubReader("t5encoder", setOf("enc", "token_embd"))

        val summary = GgufFileSummary.read(ggufFile())!!

        assertTrue(summary.components.isEmpty())
        assertFalse(summary.isAllInOne)
    }

    @Test
    fun `a reader that yields no tensors summarises to null`() {
        stubReader("sd3", emptySet())

        assertNull(GgufFileSummary.read(ggufFile()))
    }

    @Test
    fun `a failing reader summarises to null rather than throwing`() {
        GGUFReader.overrideNativeBridgeForTests {
            object : GGUFReader.NativeBridge {
                override fun getGGUFContextNativeHandle(modelPath: String): Long =
                    throw UnsatisfiedLinkError("libggufreader.so unavailable")

                override fun getContextSize(nativeHandle: Long): Long = -1L

                override fun getChatTemplate(nativeHandle: Long): String = ""

                override fun getArchitecture(nativeHandle: Long): String = ""

                override fun getParameterCount(nativeHandle: Long): String = ""

                override fun getModelName(nativeHandle: Long): String = ""

                override fun releaseGGUFContext(nativeHandle: Long) = Unit
            }
        }

        assertNull(GgufFileSummary.read(ggufFile()))
    }

    @Test
    fun `requireDiffusionOnlyGguf rejects a bundle and names what it carries`() {
        stubReader("sd3", setOf("model", "text_encoders", "first_stage_model"))

        try {
            ModelFileValidator.requireDiffusionOnlyGguf(ggufFile(), "sd3-medium-Q4_0.gguf")
            fail("Expected an all-in-one checkpoint to be rejected")
        } catch (expected: InvalidModelFileException) {
            val message = expected.message.orEmpty()
            assertTrue(message, message.contains("all-in-one"))
            assertTrue(message, message.contains("text encoders and a VAE"))
            assertTrue(message, message.contains("sd3-medium-Q4_0.gguf"))
        }
    }

    @Test
    fun `requireDiffusionOnlyGguf accepts a DiT-only checkpoint`() {
        stubReader("sd3", setOf("joint_blocks", "pos_embed"))
        val file = ggufFile()

        assertEquals(file, ModelFileValidator.requireDiffusionOnlyGguf(file))
    }

    @Test
    fun `requireDiffusionOnlyGguf accepts a file it cannot classify`() {
        stubReader("unknown", emptySet())
        val file = ggufFile()

        assertEquals(file, ModelFileValidator.requireDiffusionOnlyGguf(file))
    }

    private fun stubReader(architecture: String, prefixes: Set<String>) {
        GGUFReader.overrideNativeBridgeForTests {
            object : GGUFReader.NativeBridge {
                override fun getGGUFContextNativeHandle(modelPath: String): Long = 1L

                override fun getContextSize(nativeHandle: Long): Long = -1L

                override fun getChatTemplate(nativeHandle: Long): String = ""

                override fun getArchitecture(nativeHandle: Long): String = architecture

                override fun getParameterCount(nativeHandle: Long): String = ""

                override fun getModelName(nativeHandle: Long): String = ""

                override fun getTensorNamePrefixes(nativeHandle: Long): String =
                    prefixes.joinToString("\n")

                override fun releaseGGUFContext(nativeHandle: Long) = Unit
            }
        }
    }

    /** `GGUFReader.load` validates the GGUF magic before ever reaching the bridge. */
    private fun ggufFile(): File =
        temporaryFolder.newFile("model-${counter++}.gguf").apply {
            writeBytes("GGUF".toByteArray(Charsets.US_ASCII) + byteArrayOf(3, 0, 0, 0))
        }
}
