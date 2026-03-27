package io.aatricks.llmedge

import android.content.Context
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.runtime.SmolLM
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.*
import org.junit.Assume
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Linux-host end-to-end test for text inference using a real native library (libsmollm.so)
 * and a local model path.
 *
 * Requirements to run:
 * - Build the native smollm library for Linux and place as llmedge/build/native/linux-x86_64/libsmollm.so
 *   (Use scripts/build_smollm_linux.sh).
 * - Provide a test model path via environment variable LLMEDGE_TEST_TEXT_MODEL_PATH.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class TextInferenceLinuxE2ETest {

    private val MODEL_PATH_ENV = "LLMEDGE_TEST_TEXT_MODEL_PATH"
    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH" // Reusing this for check, though strictly not needed if LD_LIBRARY_PATH is set

    @Before
    fun resetNativeBridges() {
        SmolLM.resetNativeBridgeForTests()
        GGUFReader.resetNativeBridgeForTests()
    }

    @After
    fun tearDown() {
        SmolLM.resetNativeBridgeForTests()
        GGUFReader.resetNativeBridgeForTests()
    }

    @Test
    fun `desktop end-to-end text inference`() = runBlocking {
        // Skip test if model path is not provided
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        println("[TextInferenceLinuxE2ETest] modelPath=$modelPath")
        Assume.assumeTrue("No text test model specified in $MODEL_PATH_ENV", !modelPath.isNullOrBlank())

        // Basic check for native lib existence (informational/fail-fast)
        val projectDir = System.getProperty("user.dir")
        val defaultLibPath = "$projectDir/llmedge/build/native/linux-x86_64/libsmollm.so"
        val libPath = System.getenv(LIB_PATH_ENV) ?: System.getProperty(LIB_PATH_ENV) ?: defaultLibPath
        val libFile = File(libPath)
        if (!libFile.exists()) {
             println("[TextInferenceLinuxE2ETest] Warning: $libPath does not exist. Test might fail if not in java.library.path")
        }

        // Use Robolectric context (though SmolLM might not strictly need it for load, loadFromHuggingFace does)
        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context

        println("[TextInferenceLinuxE2ETest] Loading SmolLM model...")
        val smol = SmolLM(useVulkan = false) // Disable Vulkan for host test stability
        
        try {
            val params = SmolLM.InferenceParams(
                contextSize = 2048,
                temperature = 0.7f,
                storeChats = true
            )
            
            smol.load(modelPath!!, params)
            println("[TextInferenceLinuxE2ETest] Model loaded.")

            // Test simple generation
            val prompt = "Hello, how are you?"
            println("[TextInferenceLinuxE2ETest] Sending prompt: $prompt")
            
            smol.addUserMessage(prompt)
            val response = smol.getResponse(prompt, maxTokens = 50)
            
            println("[TextInferenceLinuxE2ETest] Response: $response")
            
            assertTrue("Response should not be empty", response.isNotEmpty())
            
            // Basic chat history check
            smol.addAssistantMessage(response)
            smol.addUserMessage("What did I just ask you?")
            val response2 = smol.getResponse("What did I just ask you?", maxTokens = 50)
            println("[TextInferenceLinuxE2ETest] Response 2: $response2")
            assertTrue("Response 2 should not be empty", response2.isNotEmpty())

        } catch (e: Exception) {
            println("[TextInferenceLinuxE2ETest] Test failed: ${e.message}")
            e.printStackTrace()
            throw e
        } finally {
            smol.close()
        }
    }
}
