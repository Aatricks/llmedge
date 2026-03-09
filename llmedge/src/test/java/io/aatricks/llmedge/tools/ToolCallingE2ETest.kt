package io.aatricks.llmedge.tools

import android.content.Context
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import java.io.File

/**
 * End-to-end test for the Tool Calling abstraction.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ToolCallingE2ETest {

    private val MODEL_PATH_ENV = "LLMEDGE_TEST_TEXT_MODEL_PATH"
    private val TOOL_MODEL_PATH_ENV = "LLMEDGE_TEST_TOOL_MODEL_PATH"

    private fun resolveToolModelPath(): String? {
        val explicit = System.getenv(TOOL_MODEL_PATH_ENV) ?: System.getProperty(TOOL_MODEL_PATH_ENV)
        if (!explicit.isNullOrBlank()) {
            return explicit
        }

        val preferredLocal = File("models/SmolLM2-1.7B-Instruct-Q8_0.gguf")
        return preferredLocal.takeIf(File::exists)?.path
    }

    @Test
    fun `test agent tool calling end-to-end`() = runBlocking {
        // 1. Tool calling needs a stronger instruct model than the tiny smoke-test model.
        val modelPath = resolveToolModelPath()
        println("[ToolCallingE2ETest] modelPath=$modelPath")
        Assume.assumeTrue(
            "Tool calling E2E requires $TOOL_MODEL_PATH_ENV or a local models/SmolLM2-1.7B-Instruct-Q8_0.gguf",
            !modelPath.isNullOrBlank() && File(modelPath).exists()
        )
        val resolvedModelPath = modelPath!!

        // 2. Initialize SmolLM
        println("[ToolCallingE2ETest] Initializing SmolLM...")
        val smol = SmolLM(useVulkan = false)
        
        try {
            smol.load(resolvedModelPath, SmolLM.InferenceParams(
                contextSize = 2048,
                temperature = 0.0f, // Use low temperature for deterministic tool calls
                storeChats = true
            ))
            println("[ToolCallingE2ETest] Model loaded.")

            // 3. Define real tools using the factory
            val factory = DeviceToolFactory(RuntimeEnvironment.getApplication())
            val batteryTool = factory.createGetBatteryStatusTool()
            var toolCalled = false
            
            // Wrap the tool to track if it was called
            val wrappedBatteryTool = batteryTool.copy(execute = { args ->
                toolCalled = true
                batteryTool.execute(args)
            })

            // 4. Create the agent
            val agent = LLMAgent(smol, listOf(wrappedBatteryTool))

            // 5. Run the chat
            val query = "What is my current battery level?"
            println("[ToolCallingE2ETest] Query: $query")
            
            val finalAnswer = agent.chat(query, maxSteps = 2)
            
            println("[ToolCallingE2ETest] Final Answer: $finalAnswer")

            // 6. Assertions
            assertTrue("Tool should have been called", toolCalled)
            assertTrue("Final answer should contain battery level", 
                finalAnswer.contains("battery", ignoreCase = true) || 
                finalAnswer.contains("%")
            )

        } catch (e: Exception) {
            println("[ToolCallingE2ETest] Test failed: ${e.message}")
            e.printStackTrace()
            throw e
        } finally {
            smol.close()
        }
    }
}
