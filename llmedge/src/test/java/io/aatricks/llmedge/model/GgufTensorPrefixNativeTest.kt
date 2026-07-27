package io.aatricks.llmedge.model

import io.aatricks.llmedge.runtime.GGUFReader
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * Exercises the real `llmedge_gguf_get_tensor_name_prefixes` against ggml's own GGUF parser,
 * which validates tensor offsets and data size — the DEPTH CONTRACT the Kotlin prefix keys depend
 * on cannot be verified by a stubbed bridge, because a stub returns whatever the test imagined.
 *
 * Opt-in: needs a desktop JNI build (`scripts/build_native_linux.sh smollm`) and fixtures from
 * `scripts/testdata/make_gguf.py`. Run with:
 *
 * ```
 * LLMEDGE_TEST_NATIVE_LIB=<path>/libsmollm.so LLMEDGE_TEST_GGUF_DIR=<dir> \
 *   ./gradlew :llmedge:testDebugUnitTest --tests "*GgufTensorPrefixNativeTest"
 * ```
 */
class GgufTensorPrefixNativeTest {
    companion object {
        private val nativeLib: String? = System.getenv("LLMEDGE_TEST_NATIVE_LIB")
        private val ggufDir: String? = System.getenv("LLMEDGE_TEST_GGUF_DIR")

        @JvmStatic
        @BeforeClass
        fun loadNativeLibrary() {
            val lib = nativeLib ?: return
            System.setProperty("llmedge.disableNativeLoad", "false")
            System.load(File(lib).absolutePath)
        }
    }

    @Test
    fun `an all-in-one bundle yields depth-1 prefixes and classifies as all-in-one`() {
        val summary = summarize("bundle.gguf")

        // Depth contract: first segment only. "model.diffusion_model.joint_blocks.0.weight"
        // must arrive as "model", never "model.diffusion_model".
        assertEquals(setOf("model", "text_encoders", "first_stage_model"), summary.tensorPrefixes)
        assertEquals("sd3", summary.architecture)
        assertTrue(summary.isAllInOne)
        assertEquals(
            setOf(GgufComponent.DIFFUSION, GgufComponent.TEXT_ENCODER, GgufComponent.VAE),
            summary.components,
        )
    }

    @Test
    fun `a DiT-only checkpoint is not an all-in-one`() {
        val summary = summarize("dit.gguf")

        // "pos_embed" has no separator, so it survives whole — the other half of the contract.
        assertEquals(setOf("joint_blocks", "pos_embed", "x_embedder"), summary.tensorPrefixes)
        assertEquals(setOf(GgufComponent.DIFFUSION), summary.components)
        assertFalse(summary.isAllInOne)
    }

    private fun summarize(fixture: String): GgufFileSummary {
        assumeTrue("Set LLMEDGE_TEST_NATIVE_LIB and LLMEDGE_TEST_GGUF_DIR", nativeLib != null && ggufDir != null)
        val file = File(ggufDir, fixture)
        assumeTrue("Missing fixture ${file.absolutePath}", file.isFile)
        GGUFReader.resetNativeBridgeForTests()
        return requireNotNull(GgufFileSummary.read(file)) { "Native reader returned no summary for $fixture" }
    }
}
