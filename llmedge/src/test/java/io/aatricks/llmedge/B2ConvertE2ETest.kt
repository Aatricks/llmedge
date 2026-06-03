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
 * Desktop-host E2E for the on-device safetensors → GGUF converter (Track B / Phase B2, Layers 5-6).
 *
 * Drives the REAL [DefaultModelRepository.resolve] with a [ModelSpec.safetensorsLocal] spec: it runs
 * the native converter (nativeConvertSafetensors) on a local HF model dir, caches the GGUF, then loads
 * and generates from it — proving the whole pipeline (convert → bake tokenizer → [quantize] → cache →
 * load → tokenize → generate) works end-to-end, not just that the GGUF KVs match a reference.
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

    /** Resolve [stDir] through the real converter at [precision], then load + greedily generate. */
    private fun convertAndGenerate(precision: ConversionPrecision): Pair<File, String> {
        val stDir = env(ST_DIR_ENV)
        Assume.assumeTrue("No safetensors dir in $ST_DIR_ENV", !stDir.isNullOrBlank())
        Assume.assumeTrue("$stDir is not a dir with model.safetensors", File(stDir, "model.safetensors").isFile)

        DesktopNativeTestSupport.requireEnabled()
        val libPath = env(LIB_PATH_ENV)
        Assume.assumeTrue("No native lib in $LIB_PATH_ENV", !libPath.isNullOrBlank())
        DesktopNativeTestSupport.requireAndLoadLibrary(libPath!!)

        val pre = env(PRE_ENV) ?: "smollm"
        val context = ApplicationProvider.getApplicationContext<Context>()

        val spec = ModelSpec.safetensorsLocal(path = stDir!!, precision = precision, tokenizerPre = pre)
        val gguf = runBlocking { DefaultModelRepository().resolve(context, spec) }
        println("[B2ConvertE2ETest] precision=$precision GGUF=${gguf.absolutePath} size=${gguf.length()}")

        assertTrue("Converted GGUF missing/empty", gguf.isFile && gguf.length() > 0L)
        assertTrue("Expected converted cache path, got ${gguf.absolutePath}", gguf.absolutePath.contains("llmedge-converted"))

        val smol = SmolLM(useVulkan = false)
        return runBlocking {
            try {
                smol.load(
                    gguf.absolutePath,
                    SmolLM.InferenceParams(contextSize = 2048, temperature = 0.0f, storeChats = true),
                )
                val prompt = "The capital of France is"
                smol.addUserMessage(prompt)
                val response = smol.getResponse(prompt, maxTokens = 16)
                println("[B2ConvertE2ETest] precision=$precision response='$response'")
                gguf to response
            } finally {
                smol.close()
            }
        }
    }

    @Test
    fun `converts a local safetensors dir on-device (f16) then loads and generates`() {
        val (_, response) = convertAndGenerate(ConversionPrecision.F16)
        val cleaned = response.replace(Regex("<\\|[^|]*\\|>"), "").trim()
        assertTrue("Expected real generated text, got: '$response'", cleaned.length >= 3)
    }

    @Test
    fun `converts then quantizes to q4_k_m on-device and generates`() {
        val stDir = env(ST_DIR_ENV)
        val (gguf, response) = convertAndGenerate(ConversionPrecision.Q4_K_M)
        // Quantized output must be materially smaller than the source safetensors.
        val sourceBytes = File(stDir!!, "model.safetensors").length()
        assertTrue(
            "Q4_K_M GGUF (${gguf.length()}) should be < source safetensors ($sourceBytes)",
            gguf.length() in 1 until sourceBytes,
        )
        val cleaned = response.replace(Regex("<\\|[^|]*\\|>"), "").trim()
        assertTrue("Expected real generated text from quantized model, got: '$response'", cleaned.length >= 3)
    }
}
