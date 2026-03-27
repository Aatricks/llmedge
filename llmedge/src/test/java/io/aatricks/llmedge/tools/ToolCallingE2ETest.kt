package io.aatricks.llmedge.tools

import io.aatricks.llmedge.LLMEdge
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.TextModelOptions
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.*
import org.junit.Assume
import org.junit.Before
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

    private fun resolveToolModelPath(): String? {
        val explicit = System.getenv(TOOL_MODEL_PATH_ENV) ?: System.getProperty(TOOL_MODEL_PATH_ENV)
        if (!explicit.isNullOrBlank()) {
            return explicit
        }
        return null
    }

    @Test
    fun `test agent tool calling end-to-end`() = runBlocking {
        val modelPath = resolveToolModelPath()
        println("[ToolCallingE2ETest] modelPath=$modelPath")
        Assume.assumeTrue(
            "Tool calling E2E requires $TOOL_MODEL_PATH_ENV to point at a tool-capable GGUF",
            !modelPath.isNullOrBlank() && File(modelPath).exists()
        )
        val resolvedModelPath = modelPath!!
        val edge = LLMEdge.create(RuntimeEnvironment.getApplication(), this)
        try {
            var toolCalled = false
            val probeTool =
                Tool(
                    name = "echo_status",
                    description = "Returns the exact status text 'status ok'.",
                    handler = {
                        toolCalled = true
                        ToolResult.success(
                            text = "status ok",
                            data = buildJsonObject { put("status", "ok") },
                        )
                    },
                )
            val agent =
                edge.text.toolAgent(
                    model = ModelSpec.localFile(resolvedModelPath),
                    options =
                        TextModelOptions(
                            contextSize = 1024,
                            temperature = 0.0f,
                            useVulkan = false,
                        ),
                    tools = listOf(probeTool),
                )
            val query = """Output exactly this tool call JSON and nothing else: {"tool":"echo_status","arguments":{}}"""
            println("[ToolCallingE2ETest] Query: $query")

            val finalAnswer = agent.reply(query, maxSteps = 2, maxTokens = 32)

            println("[ToolCallingE2ETest] Final Answer: ${finalAnswer.text}")
            assertTrue("Tool should have been called", toolCalled)
            assertTrue(
                "Final answer should contain tool result",
                finalAnswer.text.contains("status ok", ignoreCase = true),
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
