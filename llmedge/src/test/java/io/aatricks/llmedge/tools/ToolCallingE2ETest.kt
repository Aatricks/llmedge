package io.aatricks.llmedge.tools

import android.content.Context
import io.aatricks.llmedge.SmolLM
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
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

            // 3. Define a mock tool
            var toolCalled = false
            val weatherTool = Tool(
                name = "get_weather",
                description = "Get the current weather for a specific location.",
                parameters = mapOf(
                    "location" to ParameterDescription("string", "The city and state, e.g., San Francisco, CA")
                ),
                execute = { args ->
                    toolCalled = true
                    val location = args["location"] as? String ?: "Unknown"
                    println("[ToolCallingE2ETest] Tool executed for location: $location")
                    "The weather in $location is 72 degrees and sunny."
                }
            )

            // 4. Create the agent
            val agent = LLMAgent(smol, listOf(weatherTool))

            // 5. Run the chat
            val query = "Check the weather in London right now."
            println("[ToolCallingE2ETest] Query: $query")
            
            val finalAnswer = agent.chat(query, maxSteps = 2)
            
            println("[ToolCallingE2ETest] Final Answer: $finalAnswer")

            // 6. Assertions
            assertTrue("Tool should have been called", toolCalled)
            assertTrue("Final answer should contain the weather info", finalAnswer.contains("72 degrees") || finalAnswer.contains("sunny"))
            assertTrue("Final answer should mention London", finalAnswer.contains("London", ignoreCase = true))

        } catch (e: Exception) {
            println("[ToolCallingE2ETest] Test failed: ${e.message}")
            e.printStackTrace()
            throw e
        } finally {
            smol.close()
        }
    }
}
