package io.aatricks.llmedge.tools

import android.content.Context
import io.aatricks.llmedge.SmolLM
import kotlinx.coroutines.runBlocking

/**
 * Example usage of the LLMAgent with real-world device tools.
 */
object ExampleUsage {

    /**
     * Demonstrates how to initialize the agent with real-world device tools.
     * Note: In a real Android app, this would be called from a ViewModel or Activity
     * where the 'context' and 'smolLM' are available.
     */
    fun setupAgent(context: Context, smolLM: SmolLM): LLMAgent {
        // 1. Initialize the DeviceToolFactory
        val factory = DeviceToolFactory(context)
        
        // 2. Create the list of real tools
        val tools = listOf(
            factory.createGetTimeTool(),
            factory.createGetBatteryStatusTool(),
            factory.createGetDeviceInfoTool(),
            factory.createOpenBrowserTool()
        )

        // 3. Create the agent with the SmolLM instance and tools
        return LLMAgent(smolLM, tools)
    }

    // Example of running the agent
    fun runExample(context: Context, smolLM: SmolLM) = runBlocking {
        val agent = setupAgent(context, smolLM)
        
        // The LLM can now answer questions like:
        // "What time is it and how much battery do I have left?"
        // "Open google.com for me"
        val response = agent.chat("What's my current battery level?")
        println("Agent Response: $response")
    }
}
