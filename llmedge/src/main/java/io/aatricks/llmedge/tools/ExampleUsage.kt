package io.aatricks.llmedge.tools

import io.aatricks.llmedge.SmolLM
import kotlinx.coroutines.runBlocking

/**
 * Example usage of the LLMAgent with a custom tool.
 */
object ExampleUsage {

    @JvmStatic
    fun main(args: Array<String>) = runBlocking {
        // 1. Define a tool
        val weatherTool = Tool(
            name = "get_weather",
            description = "Get the current weather for a specific location.",
            parameters = mapOf(
                "location" to ParameterDescription("string", "The city and state, e.g., San Francisco, CA")
            ),
            execute = { args ->
                val location = args["location"] as? String ?: "Unknown"
                // Simulate network call
                "The weather in $location is 72 degrees and sunny."
            }
        )

        // 2. Initialize the LLM (in a real app, this would be a loaded SmolLM instance)
        // val smolLM = SmolLM()
        // smolLM.load(...)
        
        /*
        val agent = LLMAgent(smolLM, listOf(weatherTool))
        
        // 3. Chat with the agent
        val finalAnswer = agent.chat("What's the weather in London?")
        println("Final Answer: $finalAnswer")
        */
    }
}
