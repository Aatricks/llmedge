package io.aatricks.llmedge

import io.aatricks.llmedge.model.ModelChatTemplates
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.runtime.SmolLM
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
 * Desktop-host E2E proving the bundled BitNet b1.58 preset chat template ([ModelChatTemplates.BITNET])
 * produces coherent generation, whereas the GGUF's embedded template does not.
 *
 * Gated like the other LinuxE2E tests: point LLMEDGE_TEST_TEXT_MODEL_PATH at a BitNet IQ2_BN GGUF and
 * LLMEDGE_BUILD_NATIVE_LIB_PATH at a host libsmollm.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class BitNetTemplateE2ETest {
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_TEXT_MODEL_PATH"
    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"

    @Before fun reset() { SmolLM.resetNativeBridgeForTests(); GGUFReader.resetNativeBridgeForTests() }
    @After fun tearDown() { SmolLM.resetNativeBridgeForTests(); GGUFReader.resetNativeBridgeForTests() }

    @Test
    fun `bitnet generates coherent text with the preset template`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        Assume.assumeTrue("No BitNet model in $MODEL_PATH_ENV", !modelPath.isNullOrBlank())
        Assume.assumeTrue("Model is not a bitnet gguf", modelPath!!.contains("bitnet", ignoreCase = true))

        DesktopNativeTestSupport.requireEnabled()
        val libPath = System.getenv(LIB_PATH_ENV) ?: System.getProperty(LIB_PATH_ENV)
        Assume.assumeTrue("No native lib in $LIB_PATH_ENV", !libPath.isNullOrBlank())
        DesktopNativeTestSupport.requireAndLoadLibrary(libPath!!)

        val smol = SmolLM(useVulkan = false)
        try {
            smol.load(
                modelPath,
                SmolLM.InferenceParams(
                    contextSize = 2048,
                    temperature = 0.0f, // greedy → deterministic
                    storeChats = true,
                    chatTemplate = ModelChatTemplates.BITNET,
                ),
            )
            val prompt = "What is the capital of France? Answer in one word."
            smol.addUserMessage(prompt)
            val response = smol.getResponse(prompt, maxTokens = 24)
            println("[BitNetTemplateE2ETest] response='$response'")

            val cleaned = response.replace(Regex("<\\|[^|]*\\|>"), "").trim()
            assertTrue("Expected coherent text, got special-token-only: '$response'", cleaned.length >= 3)
            assertTrue(
                "Expected 'Paris' in the answer, got: '$response'",
                cleaned.contains("Paris", ignoreCase = true),
            )
        } finally {
            smol.close()
        }
    }
}
