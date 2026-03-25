package io.aatricks.llmedge.tools

import io.aatricks.llmedge.LLMEdge
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.TextModelOptions
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
        val modelPath = resolveToolModelPath()
        println("[ToolCallingE2ETest] modelPath=$modelPath")
        Assume.assumeTrue(
            "Tool calling E2E requires $TOOL_MODEL_PATH_ENV or a local models/SmolLM2-1.7B-Instruct-Q8_0.gguf",
            !modelPath.isNullOrBlank() && File(modelPath).exists()
        )
        val resolvedModelPath = modelPath!!
        val edge = LLMEdge.create(RuntimeEnvironment.getApplication(), this)
        try {
            val factory = DeviceToolFactory(RuntimeEnvironment.getApplication())
            val batteryTool = factory.createGetBatteryStatusTool()
            var toolCalled = false

            val wrappedBatteryTool = batteryTool.copy(handler = { args ->
                toolCalled = true
                batteryTool.handler(args)
            })
            val agent =
                edge.text.toolAgent(
                    model = ModelSpec.localFile(resolvedModelPath),
                    options = TextModelOptions(contextSize = 2048, temperature = 0.0f, useVulkan = false),
                    tools = listOf(wrappedBatteryTool),
                )
            val query = "What is my current battery level?"
            println("[ToolCallingE2ETest] Query: $query")

            val finalAnswer = agent.reply(query, maxSteps = 2)

            println("[ToolCallingE2ETest] Final Answer: ${finalAnswer.text}")
            assertTrue("Tool should have been called", toolCalled)
            assertTrue(
                "Final answer should contain battery level",
                finalAnswer.text.contains("battery", ignoreCase = true) ||
                    finalAnswer.text.contains("%"),
            )
        } catch (e: Exception) {
            println("[ToolCallingE2ETest] Test failed: ${e.message}")
            e.printStackTrace()
            throw e
        } finally {
            edge.close()
        }
    }
}
