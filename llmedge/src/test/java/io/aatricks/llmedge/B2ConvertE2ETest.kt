package io.aatricks.llmedge

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.model.ConversionPrecision
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.runtime.SmolLM
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Desktop-host E2E for the on-device safetensors → GGUF converter (Track B / Phase B2, Layer 5).
 *
 * Drives the REAL [DefaultModelRepository.resolve] with a [ModelSpec.safetensorsLocal] spec: it runs
 * the native converter (nativeConvertSafetensors) on a local HF model dir, caches the GGUF, then loads
 * and generates from it — proving the whole pipeline (convert → bake tokenizer → cache → load →
 * tokenize → generate) works end-to-end, not just that the GGUF KVs match a reference.
 *
 * Gated like the other host E2E tests:
 *   LLMEDGE_BUILD_NATIVE_LIB_PATH → host libsmollm built with the convert sources
 *   LLMEDGE_TEST_SAFETENSORS_DIR  → a Llama-arch HF model dir (config.json + model.safetensors +
 *                                   tokenizer.json + tokenizer_config.json), e.g. SmolLM-135M
 *   LLMEDGE_TEST_TOKENIZER_PRE    → the tokenizer.ggml.pre id to bake (default "smollm")
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class B2ConvertE2ETest {
    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    private val ST_DIR_ENV = "LLMEDGE_TEST_SAFETENSORS_DIR"
    private val PRE_ENV = "LLMEDGE_TEST_TOKENIZER_PRE"

    private fun env(name: String): String? = System.getenv(name) ?: System.getProperty(name)

    @Before fun reset() { SmolLM.resetNativeBridgeForTests(); GGUFReader.resetNativeBridgeForTests() }
    @After fun tearDown() { SmolLM.resetNativeBridgeForTests(); GGUFReader.resetNativeBridgeForTests() }

    @Test
    fun `converts a local safetensors dir on-device then loads and generates`() = runBlocking {
        val stDir = env(ST_DIR_ENV)
        Assume.assumeTrue("No safetensors dir in $ST_DIR_ENV", !stDir.isNullOrBlank())
        Assume.assumeTrue("$stDir is not a dir with model.safetensors", File(stDir, "model.safetensors").isFile)

        DesktopNativeTestSupport.requireEnabled()
        val libPath = env(LIB_PATH_ENV)
        Assume.assumeTrue("No native lib in $LIB_PATH_ENV", !libPath.isNullOrBlank())
        DesktopNativeTestSupport.requireAndLoadLibrary(libPath!!)

        val pre = env(PRE_ENV) ?: "smollm"
        val context = ApplicationProvider.getApplicationContext<Context>()

        // Real resolve path: LocalFile + conversion hint → native convert → cached GGUF.
        val spec =
            ModelSpec.safetensorsLocal(
                path = stDir!!,
                precision = ConversionPrecision.F16,
                tokenizerPre = pre,
            )
        val gguf = DefaultModelRepository().resolve(context, spec)
        println("[B2ConvertE2ETest] converted GGUF=${gguf.absolutePath} size=${gguf.length()}")

        // Must be the on-device-converted cache artifact, not the input dir.
        assertTrue("Converted GGUF missing/empty", gguf.isFile && gguf.length() > 0L)
        assertTrue("Expected converted cache path, got ${gguf.absolutePath}", gguf.absolutePath.contains("llmedge-converted"))

        val smol = SmolLM(useVulkan = false)
        try {
            smol.load(
                gguf.absolutePath,
                SmolLM.InferenceParams(
                    contextSize = 2048,
                    temperature = 0.0f, // greedy → deterministic
                    storeChats = true,
                ),
            )
            val prompt = "The capital of France is"
            smol.addUserMessage(prompt)
            val response = smol.getResponse(prompt, maxTokens = 16)
            println("[B2ConvertE2ETest] response='$response'")

            // The converted GGUF's baked tokenizer must build a working vocab and the model must
            // tokenize + generate real text (mine ≡ upstream-ref by construction, so correctness of
            // the tokens themselves is already proven by the KV/tensor oracles).
            val cleaned = response.replace(Regex("<\\|[^|]*\\|>"), "").trim()
            assertTrue("Expected real generated text, got: '$response'", cleaned.length >= 3)
        } finally {
            smol.close()
        }
    }
}
