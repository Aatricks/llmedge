package io.aatricks.llmedge.tools

import android.content.Context
import io.aatricks.llmedge.SmolLM
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

    @Test
    fun `test agent tool calling end-to-end`() = runBlocking {
        // 1. Check for model
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV) ?: "models/SmolLM2-135M-Instruct-Q8_0.gguf"
        println("[ToolCallingE2ETest] modelPath=$modelPath")
        Assume.assumeTrue("Test model not found at $modelPath", File(modelPath).exists())

        // 2. Initialize SmolLM
        println("[ToolCallingE2ETest] Initializing SmolLM...")
        val smol = SmolLM(useVulkan = false)
        
        try {
            smol.load(modelPath, SmolLM.InferenceParams(
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
