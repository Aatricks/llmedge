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

    /**
     * The published SD3 Medium checkpoints, which is what an importer actually picks up. Only the
     * header is needed, so the fixtures are HTTP range reads rather than multi-GB downloads —
     * `gguf_init_from_file` with `no_alloc` accepts a file truncated after the tensor infos:
     *
     * ```
     * curl -L -r 0-8388607 -o real-sd3-dit.gguf \
     *   https://huggingface.co/city96/stable-diffusion-3-medium-gguf/resolve/main/sd3_medium-Q4_0.gguf
     * curl -L -r 0-8388607 -o real-sd3-bundle.gguf \
     *   https://huggingface.co/second-state/stable-diffusion-3-medium-GGUF/resolve/main/sd3-medium-Q4_0.gguf
     * ```
     */
    @Test
    fun `the published city96 SD3 DiT is classified as diffusion-only`() {
        val summary = summarize("real-sd3-dit.gguf")

        println("city96 sd3_medium-Q4_0 prefixes: ${summary.tensorPrefixes.sorted()}")
        assertTrue(summary.components.toString(), GgufComponent.DIFFUSION in summary.components)
        assertFalse("must not be rejected for a diffusion-only preset", summary.isAllInOne)
    }

    @Test
    fun `the published second-state SD3 bundle is classified as all-in-one`() {
        val summary = summarize("real-sd3-bundle.gguf")

        println("second-state sd3-medium-Q4_0 prefixes: ${summary.tensorPrefixes.sorted()}")
        assertTrue("must be rejected for a diffusion-only preset", summary.isAllInOne)
    }

    private fun summarize(fixture: String): GgufFileSummary {
        assumeTrue("Set LLMEDGE_TEST_NATIVE_LIB and LLMEDGE_TEST_GGUF_DIR", nativeLib != null && ggufDir != null)
        val file = File(ggufDir, fixture)
        assumeTrue("Missing fixture ${file.absolutePath}", file.isFile)
        GGUFReader.resetNativeBridgeForTests()
        return requireNotNull(GgufFileSummary.read(file)) { "Native reader returned no summary for $fixture" }
    }
}
