package io.aatricks.llmedge.tools

import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject

/**
 * An Agent runner that wraps [SmolLM] to provide automatic tool calling (Function Calling).
 *
 * @property smol The [SmolLM] instance used for inference.
 * @property tools The list of [Tool] available to the LLM.
 */
class LLMAgent(
    private val smol: SmolLM,
    private val tools: List<Tool>
) {

    /**
     * Processes a user message, potentially invoking tools, until a final text answer is reached
     * or the [maxSteps] limit is hit.
     *
     * @param userMessage The initial input from the user.
     * @param maxSteps The maximum number of tool-call iterations to allow to prevent infinite loops.
     * @return The final text response from the LLM.
     */
    suspend fun chat(userMessage: String, maxSteps: Int = 3): String = withContext(Dispatchers.IO) {
        // Step a: Inject the system prompt with tool definitions
        val systemPrompt = ToolPromptGenerator.generateSystemPrompt(tools)
        smol.addSystemPrompt(systemPrompt)

        // Step b: Add the user message
        smol.addUserMessage(userMessage)

        var currentStep = 0
        var lastResponse = ""

        while (currentStep < maxSteps) {
            // Step c: Call model. Passing an empty string for the query because 
            // the context was built via addSystemPrompt / addUserMessage.
            val response = smol.getResponse("")
            smol.addAssistantMessage(response)
            lastResponse = response

            // Step d: Check for tool call signature
            val jsonObject = JsonExtractor.extractToolCallJson(response)

            if (jsonObject != null && jsonObject.has("tool_name")) {
                // Step f: Execute tool
                val toolName = jsonObject.getString("tool_name")
                val tool = tools.find { it.name == toolName }
                
                if (tool != null) {
                    val arguments = try {
                        val argsObj = jsonObject.optJSONObject("arguments") ?: JSONObject()
                        val argsMap = mutableMapOf<String, Any>()
                        val keys = argsObj.keys()
                        while (keys.hasNext()) {
                            val key = keys.next()
                            argsMap[key] = argsObj.get(key)
                        }
                        argsMap
                    } catch (e: JSONException) {
                        emptyMap<String, Any>()
                    }

                    try {
                        val result = tool.execute(arguments)
                        smol.addUserMessage("TOOL_RESULT for $toolName: $result. Use this to provide the final response to the user.")
                    } catch (e: Exception) {
                        smol.addUserMessage("TOOL_RESULT for $toolName: Error executing tool - ${e.message}")
                    }
                } else {
                    smol.addUserMessage("TOOL_RESULT: Error - Tool '$toolName' not found.")
                }
            } else {
                // Step e: No tool call detected, assume this is the final answer
                return@withContext response
            }

            currentStep++
        }

        // Reached max steps without final answer
        return@withContext lastResponse
    }
}
