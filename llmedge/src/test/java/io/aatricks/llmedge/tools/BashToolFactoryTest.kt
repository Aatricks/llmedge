package io.aatricks.llmedge.tools

import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BashToolFactoryTest {
    @Test
    fun `bash tool requires exactly one of argv or command`() = runTest {
        val tool = BashToolFactory().createBashTool()

        val missing = tool.handler(buildJsonObject { })
        val both =
            tool.handler(
                buildJsonObject {
                    put("command", "pwd")
                    put("argv", buildJsonArray { add(JsonPrimitive("pwd")) })
                },
            )

        assertTrue(missing.isError)
        assertTrue(missing.text.contains("exactly one"))
        assertTrue(both.isError)
        assertTrue(both.text.contains("exactly one"))
    }

    @Test
    fun `structured argv execution succeeds and returns structured result`() = runTest {
        val executor =
            RecordingExecutor {
                BashExecutionResult(
                    exitCode = 0,
                    stdout = "hello\n",
                    stderr = "",
                )
            }
        val tool =
            BashToolFactory.forTesting(
                options = BashToolOptions(defaultWorkingDirectory = "/repo"),
                executor = executor,
            ).createBashTool()

        val result =
            tool.handler(
                buildJsonObject {
                    put(
                        "argv",
                        buildJsonArray {
                            add(JsonPrimitive("echo"))
                            add(JsonPrimitive("hello"))
                        },
                    )
                },
            )

        assertFalse(result.isError)
        assertEquals(1, executor.requests.size)
        assertEquals(listOf("echo", "hello"), executor.requests.single().argv)
        assertNull(executor.requests.single().command)
        assertEquals("/repo", executor.requests.single().workingDirectory)
        assertEquals(0, result.data["exitCode"]?.jsonPrimitive?.intOrNull)
        assertEquals("hello\n", result.data["stdout"]?.jsonPrimitive?.contentOrNull)
        assertEquals("echo", result.data["argv"]?.jsonArray?.first()?.jsonPrimitive?.contentOrNull)
        assertTrue(result.text.contains("exit code 0"))
    }

    @Test
    fun `non zero exit is reported as an error`() = runTest {
        val tool =
            BashToolFactory.forTesting(
                options = BashToolOptions(),
                executor =
                    RecordingExecutor {
                        BashExecutionResult(
                            exitCode = 2,
                            stdout = "",
                            stderr = "bad option",
                        )
                    },
            ).createBashTool()

        val result =
            tool.handler(
                buildJsonObject {
                    put(
                        "argv",
                        buildJsonArray {
                            add(JsonPrimitive("ls"))
                            add(JsonPrimitive("--bad-option"))
                        },
                    )
                },
            )

        assertTrue(result.isError)
        assertEquals(2, result.data["exitCode"]?.jsonPrimitive?.intOrNull)
        assertTrue(result.text.contains("exit code 2"))
        assertTrue(result.text.contains("bad option"))
    }

    @Test
    fun `raw shell execution is rejected by default`() = runTest {
        val executor = RecordingExecutor { BashExecutionResult(exitCode = 0, stdout = "", stderr = "") }
        val tool =
            BashToolFactory.forTesting(
                options = BashToolOptions(),
                executor = executor,
            ).createBashTool()

        val result =
            tool.handler(
                buildJsonObject {
                    put("command", "pwd")
                },
            )

        assertTrue(result.isError)
        assertTrue(result.text.contains("disabled"))
        assertTrue(executor.requests.isEmpty())
    }

    @Test
    fun `raw shell execution succeeds when enabled`() = runTest {
        val executor =
            RecordingExecutor {
                BashExecutionResult(
                    exitCode = 0,
                    stdout = "/repo\n",
                    stderr = "",
                )
            }
        val tool =
            BashToolFactory.forTesting(
                options = BashToolOptions(allowRawShell = true, shellExecutable = "/bin/bash"),
                executor = executor,
            ).createBashTool()

        val result =
            tool.handler(
                buildJsonObject {
                    put("command", "pwd")
                    put("workingDirectory", "/tmp/demo")
                },
            )

        assertFalse(result.isError)
        assertEquals(listOf("/bin/bash", "-lc", "pwd"), executor.requests.single().argv)
        assertEquals("pwd", executor.requests.single().command)
        assertEquals("/tmp/demo", executor.requests.single().workingDirectory)
        assertEquals("pwd", result.data["command"]?.jsonPrimitive?.contentOrNull)
    }

    @Test
    fun `raw shell execution runs real bash process`() = runTest {
        val workingDirectory = System.getProperty("java.io.tmpdir")
        val tool =
            BashToolFactory(
                BashToolOptions(
                    allowRawShell = true,
                    defaultWorkingDirectory = workingDirectory,
                ),
            ).createBashTool()

        val result =
            tool.handler(
                buildJsonObject {
                    put("command", "pwd")
                },
            )

        assertFalse(result.isError)
        assertEquals(0, result.data["exitCode"]?.jsonPrimitive?.intOrNull)
        assertEquals("${workingDirectory.trimEnd('/')}\n", result.data["stdout"]?.jsonPrimitive?.contentOrNull)
        assertEquals("pwd", result.data["command"]?.jsonPrimitive?.contentOrNull)
    }

    @Test
    fun `timeouts are reported as errors`() = runTest {
        val tool =
            BashToolFactory.forTesting(
                options = BashToolOptions(),
                executor =
                    RecordingExecutor {
                        BashExecutionResult(
                            exitCode = null,
                            stdout = "",
                            stderr = "timed out",
                            timedOut = true,
                        )
                    },
            ).createBashTool()

        val result =
            tool.handler(
                buildJsonObject {
                    put(
                        "argv",
                        buildJsonArray {
                            add(JsonPrimitive("sleep"))
                            add(JsonPrimitive("10"))
                        },
                    )
                },
            )

        assertTrue(result.isError)
        assertTrue(result.data["timedOut"]?.jsonPrimitive?.booleanOrNull == true)
        assertTrue(result.text.contains("timed out"))
    }

    @Test
    fun `output is truncated to configured limit`() = runTest {
        val tool =
            BashToolFactory.forTesting(
                options = BashToolOptions(maxOutputChars = 4),
                executor =
                    RecordingExecutor {
                        BashExecutionResult(
                            exitCode = 0,
                            stdout = "abcdefgh",
                            stderr = "wxyz123",
                        )
                    },
            ).createBashTool()

        val result =
            tool.handler(
                buildJsonObject {
                    put(
                        "argv",
                        buildJsonArray {
                            add(JsonPrimitive("echo"))
                            add(JsonPrimitive("abcdefgh"))
                        },
                    )
                },
            )

        assertFalse(result.isError)
        assertEquals("abcd", result.data["stdout"]?.jsonPrimitive?.contentOrNull)
        assertEquals("wxyz", result.data["stderr"]?.jsonPrimitive?.contentOrNull)
        assertTrue(result.data["truncated"]?.jsonPrimitive?.booleanOrNull == true)
        assertTrue(result.text.contains("truncated"))
    }

    private class RecordingExecutor(
        private val block: suspend (BashExecutionRequest) -> BashExecutionResult,
    ) : BashCommandExecutor {
        val requests = mutableListOf<BashExecutionRequest>()

        override suspend fun run(request: BashExecutionRequest): BashExecutionResult {
            requests += request
            return block(request)
        }
    }
}
