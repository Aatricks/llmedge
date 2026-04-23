package io.aatricks.llmedge.tools

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdge
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.ConversationWindow
import io.aatricks.llmedge.text.TextModelOptions
import io.aatricks.llmedge.text.runtime.SmolLM
import java.io.File
import java.nio.file.Files
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ToolAgentTest {
    private fun createTempGgufFile(dir: File): File =
        File.createTempFile("llmedge-tool-agent", ".gguf", dir).apply {
            writeBytes(
                byteArrayOf(
                    'G'.code.toByte(),
                    'G'.code.toByte(),
                    'U'.code.toByte(),
                    'F'.code.toByte(),
                    0x00,
                ),
            )
        }

    @Before
    fun setUp() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        GGUFReader.overrideNativeBridgeForTests {
            object : GGUFReader.NativeBridge {
                override fun getGGUFContextNativeHandle(modelPath: String): Long = 1L

                override fun getContextSize(nativeHandle: Long): Long = 2048L

                override fun getChatTemplate(nativeHandle: Long): String =
                    "{% for message in messages %}{{ message.content }}{% endfor %}"

                override fun getArchitecture(nativeHandle: Long): String = "llama"

                override fun getParameterCount(nativeHandle: Long): String = "135M"

                override fun getModelName(nativeHandle: Long): String = "Test GGUF"

                override fun releaseGGUFContext(nativeHandle: Long) = Unit
            }
        }
    }

    @After
    fun tearDown() {
        SmolLM.resetNativeBridgeForTests()
        GGUFReader.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `tool schema rejects missing and mistyped arguments`() {
        val schema =
            ToolSchema(
                parameters =
                    mapOf(
                        "url" to ToolParameter(ToolParameterType.STRING, "url"),
                        "confirm" to ToolParameter(ToolParameterType.BOOLEAN, "confirm"),
                    ),
            )

        val missing = schema.validate(buildJsonObject { put("url", "https://example.com") })
        val mistyped =
            schema.validate(
                buildJsonObject {
                    put("url", "https://example.com")
                    put("confirm", "yes")
                },
            )

        assertTrue(missing.any { it.contains("Missing required argument 'confirm'") })
        assertTrue(mistyped.any { it.contains("confirm") })
    }

    @Test
    fun `tool schema ignores unexpected arguments for zero arg tools`() {
        val errors =
            ToolSchema().validate(
                buildJsonObject {
                    put("time", "2026-03-25")
                    put("battery", 80)
                },
            )

        assertTrue(errors.isEmpty())
    }

    @Test
    fun `tool schema rejects unexpected arguments when parameters are declared`() {
        val errors =
            ToolSchema(
                parameters =
                    mapOf("url" to ToolParameter(ToolParameterType.STRING, "url")),
            ).validate(
                buildJsonObject {
                    put("url", "https://example.com")
                    put("confirm", true)
                },
            )

        assertTrue(errors.any { it.contains("Unexpected argument 'confirm'") })
    }

    @Test
    fun `reply executes read only tool and returns final answer`() = runTest {
        installBridge(
            listOf("""{"tool":"lookup","arguments":{"query":"Paris"}}"""),
            listOf("Paris is the capital of France."),
        )
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                memory = ConversationWindow(maxTurns = 4, maxTokens = 512),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        Tool(
                            name = "lookup",
                            description = "Looks up info",
                            schema =
                                ToolSchema(
                                    parameters =
                                        mapOf("query" to ToolParameter(ToolParameterType.STRING, "query")),
                                ),
                            handler = { args ->
                                ToolResult.success(
                                    "Paris lookup complete",
                                    buildJsonObject {
                                        put("query", args["query"]?.toString()?.trim('"') ?: "")
                                    },
                                )
                            },
                        ),
                    ),
            )

        try {
            val result = agent.reply("What is the capital of France?", maxSteps = 3)

            assertEquals(ToolAgentFinishReason.COMPLETED, result.finishReason)
            assertEquals("Paris is the capital of France.", result.text)
            assertEquals(2, result.trace.size)
            assertEquals("lookup", result.trace.first().toolCall?.tool)
            assertTrue(agent.historySnapshot().last().content.contains("capital of France"))
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply executes zero arg tool when model invents extra fields`() = runTest {
        installBridge(
            listOf("""{"tool":"clock","arguments":{"time":"2026-03-25"}}"""),
            listOf("It is 10:15."),
        )
        val edge = createEdge(this)
        var invoked = false
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        Tool(
                            name = "clock",
                            description = "Returns the current time.",
                            handler = {
                                invoked = true
                                ToolResult.success("Current time is 10:15.")
                            },
                        ),
                    ),
            )

        try {
            val result = agent.reply("What time is it?", maxSteps = 2)

            assertTrue(invoked)
            assertEquals("It is 10:15.", result.text)
            assertTrue(result.trace.first().toolResult?.isError == false)
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply denies action tool by default`() = runTest {
        installBridge(
            listOf("""{"tool":"open_browser","arguments":{"url":"https://example.com"}}"""),
            listOf("I cannot open the browser without approval."),
        )
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools = listOf(actionTool()),
            )

        try {
            val result = agent.reply("Open example.com", maxSteps = 2)

            assertEquals(ToolAgentFinishReason.COMPLETED, result.finishReason)
            assertTrue(result.trace.first().toolResult?.isError == true)
            assertTrue(result.trace.first().toolDeniedReason?.contains("explicit approval") == true)
            assertTrue(result.text.contains("without approval"))
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply allows action tool with explicit policy`() = runTest {
        installBridge(
            listOf("""{"tool":"open_browser","arguments":{"url":"https://example.com"}}"""),
            listOf("Opened example.com."),
        )
        val edge = createEdge(this)
        var invoked = false
        val tool =
            actionTool().copy(
                handler = {
                    invoked = true
                    ToolResult.success("Opened example.com.", buildJsonObject { put("url", "https://example.com") })
                },
            )
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools = listOf(tool),
                policy = ToolPolicies.ALLOW_ALL,
            )

        try {
            val result = agent.reply("Open example.com", maxSteps = 2)

            assertTrue(invoked)
            assertEquals("Opened example.com.", result.text)
            assertTrue(result.trace.first().toolDeniedReason == null)
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply denies bash tool by default`() = runTest {
        installBridge(
            listOf("""{"tool":"run_bash_command","arguments":{"argv":["echo","hello"]}}"""),
            listOf("I cannot run shell commands without approval."),
        )
        val edge = createEdge(this)
        val executor = RecordingBashExecutor { BashExecutionResult(exitCode = 0, stdout = "hello\n", stderr = "") }
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        BashToolFactory.forTesting(BashToolOptions(), executor).createBashTool(),
                    ),
            )

        try {
            val result = agent.reply("Run echo hello", maxSteps = 2)

            assertTrue(result.trace.first().toolResult?.isError == true)
            assertTrue(result.trace.first().toolDeniedReason?.contains("explicit approval") == true)
            assertTrue(executor.requests.isEmpty())
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply executes bash tool with explicit policy`() = runTest {
        installBridge(
            listOf("""{"tool":"run_bash_command","arguments":{"argv":["echo","hello"]}}"""),
            listOf("The command printed hello."),
        )
        val edge = createEdge(this)
        val executor =
            RecordingBashExecutor {
                BashExecutionResult(exitCode = 0, stdout = "hello\n", stderr = "")
            }
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        BashToolFactory.forTesting(BashToolOptions(), executor).createBashTool(),
                    ),
                policy = ToolPolicies.ALLOW_ALL,
            )

        try {
            val result = agent.reply("Run echo hello", maxSteps = 2)

            assertEquals("The command printed hello.", result.text)
            assertEquals(1, executor.requests.size)
            assertEquals(listOf("echo", "hello"), executor.requests.single().argv)
            assertTrue(result.trace.first().toolResult?.data?.toString()?.contains("hello") == true)
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply executes real bash command through tool agent`() = runTest {
        val cacheDir = ApplicationProvider.getApplicationContext<Context>().cacheDir
        cacheDir.mkdirs()
        val workingDirectory = Files.createTempDirectory(cacheDir.toPath(), "bash-tool-agent-").toFile().absolutePath
        installBridge(
            listOf("""{"tool":"run_bash_command","arguments":{"command":"pwd","workingDirectory":"$workingDirectory"}}"""),
            listOf("The command returned the requested working directory."),
        )
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        BashToolFactory(
                            BashToolOptions(allowRawShell = true),
                        ).createBashTool(),
                    ),
                policy = ToolPolicies.ALLOW_ALL,
            )

        try {
            val result = agent.reply("Run pwd in the cache dir", maxSteps = 2)

            assertEquals("The command returned the requested working directory.", result.text)
            assertTrue(result.trace.first().toolResult?.isError == false)
            assertEquals(
                "$workingDirectory\n",
                result.trace.first().toolResult?.data?.get("stdout")?.jsonPrimitive?.contentOrNull,
            )
            assertEquals(
                "pwd",
                result.trace.first().toolResult?.data?.get("command")?.jsonPrimitive?.contentOrNull,
            )
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply reports unknown tool and continues`() = runTest {
        installBridge(
            listOf("""{"tool":"missing_tool","arguments":{}}"""),
            listOf("I do not have that capability."),
        )
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools = emptyList(),
            )

        try {
            val result = agent.reply("Use the missing tool", maxSteps = 2)

            assertTrue(result.trace.first().toolResult?.isError == true)
            assertTrue(result.trace.first().toolResult?.text?.contains("not registered") == true)
            assertEquals("I do not have that capability.", result.text)
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply stops at max steps when the model keeps calling tools`() = runTest {
        installBridge(
            listOf("""{"tool":"lookup","arguments":{"query":"one"}}"""),
            listOf("""{"tool":"lookup","arguments":{"query":"two"}}"""),
        )
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        Tool(
                            name = "lookup",
                            description = "Looks up info",
                            schema =
                                ToolSchema(
                                    parameters =
                                        mapOf("query" to ToolParameter(ToolParameterType.STRING, "query")),
                                ),
                            handler = { ToolResult.success("ok") },
                        ),
                    ),
            )

        try {
            val result = agent.reply("Keep looping", maxSteps = 2)

            assertEquals(ToolAgentFinishReason.MAX_STEPS, result.finishReason)
            assertEquals(2, result.trace.size)
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply strips think blocks from final answer and history`() = runTest {
        installBridge(listOf("<think>private reasoning</think>Visible answer"))
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools = emptyList(),
            )

        try {
            val result = agent.reply("Say something useful.", maxSteps = 1)

            assertEquals("Visible answer", result.text)
            assertEquals("Visible answer", agent.historySnapshot().last().content)
            assertTrue(result.trace.single().rawModelOutput.contains("<think>private reasoning</think>"))
        } finally {
            edge.close()
        }
    }

    @Test
    fun `reply retries when sanitized final text is empty`() = runTest {
        val prompts = mutableListOf<String>()
        installBridge(
            listOf("""{"tool":"lookup","arguments":{"query":"Paris"}}"""),
            listOf("<think>private reasoning only</think>"),
            listOf("Visible answer after retry"),
            promptLog = prompts,
        )
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        Tool(
                            name = "lookup",
                            description = "Looks up info",
                            schema =
                                ToolSchema(
                                    parameters =
                                        mapOf("query" to ToolParameter(ToolParameterType.STRING, "query")),
                                ),
                            handler = { ToolResult.success("Paris lookup complete") },
                        ),
                    ),
            )

        try {
            val result = agent.reply("What is Paris?", maxSteps = 3)

            assertEquals(ToolAgentFinishReason.COMPLETED, result.finishReason)
            assertEquals("Visible answer after retry", result.text)
            assertEquals(3, result.trace.size)
            assertTrue(result.trace[1].rawModelOutput.contains("<think>private reasoning only</think>"))
            assertEquals(3, prompts.size)
            assertTrue(prompts[2].contains("Your previous response contained no user-visible text"))
            assertTrue(!prompts[2].contains("<think>private reasoning only</think>"))
        } finally {
            edge.close()
        }
    }

    @Test
    fun `stream emits tool events before final text`() = runTest {
        installBridge(
            listOf("""{"tool":"lookup","arguments":{"query":"Paris"}}"""),
            listOf("Paris ", "is the capital."),
        )
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        Tool(
                            name = "lookup",
                            description = "Looks up info",
                            schema =
                                ToolSchema(
                                    parameters =
                                        mapOf("query" to ToolParameter(ToolParameterType.STRING, "query")),
                                ),
                            handler = { ToolResult.success("ok") },
                        ),
                    ),
            )

        try {
            val events = agent.stream("What is Paris?", maxSteps = 2).toList()

            assertTrue(events.first() is ToolAgentEvent.Started)
            assertTrue(events.any { it is ToolAgentEvent.ToolCallRequested })
            assertTrue(events.any { it is ToolAgentEvent.ToolExecuting })
            assertTrue(events.any { it is ToolAgentEvent.ToolResultReceived })
            assertEquals(
                "Paris is the capital.",
                events.filterIsInstance<ToolAgentEvent.TextChunk>().joinToString("") { it.value },
            )
            assertTrue(events.last() is ToolAgentEvent.Completed)
        } finally {
            edge.close()
        }
    }

    @Test
    fun `stream retries when sanitized final text is empty`() = runTest {
        val prompts = mutableListOf<String>()
        installBridge(
            listOf("""{"tool":"lookup","arguments":{"query":"Paris"}}"""),
            listOf("<think>private reasoning only</think>"),
            listOf("Visible answer after retry"),
            promptLog = prompts,
        )
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools =
                    listOf(
                        Tool(
                            name = "lookup",
                            description = "Looks up info",
                            schema =
                                ToolSchema(
                                    parameters =
                                        mapOf("query" to ToolParameter(ToolParameterType.STRING, "query")),
                                ),
                            handler = { ToolResult.success("Paris lookup complete") },
                        ),
                    ),
            )

        try {
            val events = agent.stream("What is Paris?", maxSteps = 3).toList()
            val completed = events.last() as ToolAgentEvent.Completed

            assertEquals("Visible answer after retry", completed.result.text)
            assertEquals(
                "Visible answer after retry",
                events.filterIsInstance<ToolAgentEvent.TextChunk>().joinToString("") { it.value },
            )
            assertEquals(1, events.filterIsInstance<ToolAgentEvent.ToolCallRequested>().size)
            assertTrue(completed.result.trace[1].rawModelOutput.contains("<think>private reasoning only</think>"))
            assertEquals(3, prompts.size)
            assertTrue(prompts[2].contains("Your previous response contained no user-visible text"))
            assertTrue(!prompts[2].contains("<think>private reasoning only</think>"))
        } finally {
            edge.close()
        }
    }

    @Test
    fun `stream strips think blocks before emitting final text`() = runTest {
        installBridge(listOf("<think>private reasoning</think>Visible ", "answer"))
        val edge = createEdge(this)
        val agent =
            edge.text.toolAgent(
                model = localModel(edge),
                options = TextModelOptions(temperature = 0.0f, useVulkan = false),
                tools = emptyList(),
            )

        try {
            val events = agent.stream("Answer plainly.", maxSteps = 1).toList()
            val emittedText = events.filterIsInstance<ToolAgentEvent.TextChunk>().joinToString("") { it.value }
            val completed = events.last() as ToolAgentEvent.Completed

            assertEquals("Visible answer", emittedText)
            assertEquals("Visible answer", completed.result.text)
            assertTrue(events.filterIsInstance<ToolAgentEvent.TextChunk>().none { it.value.contains("<think>") })
            assertTrue(completed.result.trace.single().rawModelOutput.contains("<think>private reasoning</think>"))
        } finally {
            edge.close()
        }
    }

    private fun createEdge(scope: kotlinx.coroutines.CoroutineScope): LLMEdge {
        val context = ApplicationProvider.getApplicationContext<Context>()
        return LLMEdge.create(context, scope)
    }

    private fun localModel(edge: LLMEdge) =
        io.aatricks.llmedge.model.ModelSpec.localFile(
            createTempGgufFile(ApplicationProvider.getApplicationContext<Context>().cacheDir),
        )

    private fun actionTool(): Tool =
        Tool(
            name = "open_browser",
            description = "Opens a URL.",
            kind = ToolKind.ACTION,
            schema =
                ToolSchema(
                    parameters =
                        mapOf("url" to ToolParameter(ToolParameterType.STRING, "url")),
                ),
            handler = { args ->
                ToolResult.success(
                    "Opened ${args["url"]}.",
                    buildJsonObject { put("url", args["url"]?.toString() ?: "") },
                )
            },
        )

    private class RecordingBashExecutor(
        private val block: suspend (BashExecutionRequest) -> BashExecutionResult,
    ) : BashCommandExecutor {
        val requests = mutableListOf<BashExecutionRequest>()

        override suspend fun run(request: BashExecutionRequest): BashExecutionResult {
            requests += request
            return block(request)
        }
    }

    private fun installBridge(
        vararg responses: List<String>,
        promptLog: MutableList<String>? = null,
    ) {
        SmolLM.overrideNativeBridgeForTests {
            object : SmolLM.NativeBridge {
                private var currentResponse = ArrayDeque<String>()
                private var responseIndex = 0

                override fun loadModel(
                    instance: SmolLM,
                    modelPath: String,
                    minP: Float,
                    temperature: Float,
                    storeChats: Boolean,
                    contextSize: Long,
                    chatTemplate: String,
                    nThreads: Int,
                    useMmap: Boolean,
                    useMlock: Boolean,
                    useVulkan: Boolean,
                    useFlashAttn: Boolean,
                    kvCacheTypeK: Int,
                    kvCacheTypeV: Int,
                    nGpuLayers: Int,
                ): Long = 1L

                override fun setReasoningOptions(
                    instance: SmolLM,
                    modelPtr: Long,
                    disableThinking: Boolean,
                    reasoningBudget: Int,
                ) = Unit

                override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) = Unit

                override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 42f

                override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 1L

                override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 1000L

                override fun clearMessages(instance: SmolLM, modelPtr: Long) = Unit

                override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 64

                override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L

                override fun nativeDecodePreparedEmbeddings(
                    instance: SmolLM,
                    modelPtr: Long,
                    embdPath: String,
                    metaPath: String,
                    nBatch: Int,
                ): Boolean = true

                override fun close(instance: SmolLM, modelPtr: Long) = Unit

                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    promptLog?.add(prompt)
                    val response = responses[responseIndex++]
                    currentResponse = ArrayDeque(response + listOf("[EOG]"))
                }

                override fun completionLoop(instance: SmolLM, modelPtr: Long): String = currentResponse.removeFirst()

                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String =
                    currentResponse.removeFirst()

                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit

                override fun clearKvCache(instance: SmolLM, modelPtr: Long) = Unit
            }
        }
    }
}
