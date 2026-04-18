package io.aatricks.llmedge.tools

import java.io.File
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.supervisorScope
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

data class BashToolOptions(
    val shellExecutable: String = "bash",
    val allowRawShell: Boolean = false,
    val defaultWorkingDirectory: String? = null,
    val environment: Map<String, String> = emptyMap(),
    val timeoutMillis: Long = 30_000,
    val maxOutputChars: Int = 8_000,
)

class BashToolFactory private constructor(
    private val options: BashToolOptions,
    private val executor: BashCommandExecutor,
) {
    @JvmOverloads
    constructor(options: BashToolOptions = BashToolOptions()) : this(options, ProcessBuilderBashCommandExecutor())

    fun createBashTool(): Tool =
        Tool(
            name = "run_bash_command",
            description =
                "Runs a shell command. Provide either argv as an array of command arguments or command as a raw shell string. " +
                    "Raw shell strings require explicit factory opt-in.",
            kind = ToolKind.ACTION,
            schema =
                ToolSchema(
                    parameters =
                        mapOf(
                            "argv" to
                                ToolParameter(
                                    type = ToolParameterType.ARRAY,
                                    description = "Structured command arguments, for example [\"echo\", \"hello\"]",
                                    required = false,
                                ),
                            "command" to
                                ToolParameter(
                                    type = ToolParameterType.STRING,
                                    description = "Raw shell command string executed via bash -lc.",
                                    required = false,
                                ),
                            "workingDirectory" to
                                ToolParameter(
                                    type = ToolParameterType.STRING,
                                    description = "Optional working directory override for this command.",
                                    required = false,
                                ),
                        ),
                ),
            handler = ::runCommand,
        )

    private suspend fun runCommand(arguments: JsonObject): ToolResult {
        val rawCommand = arguments["command"]?.jsonPrimitive?.contentOrNull
        val argvElement = arguments["argv"]
        val workingDirectory = arguments["workingDirectory"]?.jsonPrimitive?.contentOrNull ?: options.defaultWorkingDirectory

        if ((rawCommand == null) == (argvElement == null)) {
            return invalidArguments("Provide exactly one of 'argv' or 'command'.")
        }

        val argv =
            if (argvElement != null) {
                parseArgv(argvElement) ?: return invalidArguments("Argument 'argv' must be a non-empty array of strings.")
            } else {
                if (!options.allowRawShell) {
                    return ToolResult.error(
                        text = "Raw shell execution is disabled for this bash tool.",
                        data = toolErrorData("raw_shell_disabled", "Raw shell execution is disabled for this bash tool."),
                    )
                }
                val command = rawCommand?.takeUnless(String::isBlank)
                    ?: return invalidArguments("Argument 'command' must be a non-empty string.")
                listOf(options.shellExecutable, "-lc", command)
            }

        val resolvedWorkingDirectory = workingDirectory?.takeUnless(String::isBlank)

        val request =
            BashExecutionRequest(
                argv = argv,
                command = rawCommand,
                workingDirectory = resolvedWorkingDirectory,
                environment = options.environment,
                timeoutMillis = options.timeoutMillis,
                maxOutputChars = options.maxOutputChars,
            )

        val execution =
            runCatching { executor.run(request) }
                .getOrElse { error ->
                    return ToolResult.error(
                        text = "Failed to run command: ${error.message ?: "Unknown error."}",
                        data =
                            buildJsonObject {
                                put("code", "execution_failed")
                                put("message", error.message ?: "Unknown error.")
                                put("argv", request.argv.toJsonArray())
                                request.command?.let { put("command", it) }
                                request.workingDirectory?.let { put("workingDirectory", it) }
                            },
                    )
                }

        val truncatedStdout = execution.stdout.limitTo(options.maxOutputChars)
        val truncatedStderr = execution.stderr.limitTo(options.maxOutputChars)
        val truncated = truncatedStdout != execution.stdout || truncatedStderr != execution.stderr
        val data =
            buildJsonObject {
                execution.exitCode?.let { put("exitCode", it) }
                put("stdout", truncatedStdout)
                put("stderr", truncatedStderr)
                put("timedOut", execution.timedOut)
                put("truncated", truncated)
                put("argv", request.argv.toJsonArray())
                request.command?.let { put("command", it) }
                request.workingDirectory?.let { put("workingDirectory", it) }
            }

        val text =
            buildString {
                if (execution.timedOut) {
                    append("Command timed out after ${request.timeoutMillis} ms.")
                } else {
                    append("Command completed with exit code ${execution.exitCode ?: "unknown"}.")
                }
                if (truncatedStdout.isNotEmpty()) {
                    append(" Stdout: ")
                    append(truncatedStdout)
                }
                if (truncatedStderr.isNotEmpty()) {
                    append(" Stderr: ")
                    append(truncatedStderr)
                }
                if (truncated) {
                    append(" Output was truncated.")
                }
            }

        val isError = execution.timedOut || (execution.exitCode != null && execution.exitCode != 0)
        return if (isError) ToolResult.error(text = text, data = data) else ToolResult.success(text = text, data = data)
    }

    private fun invalidArguments(message: String): ToolResult =
        ToolResult.error(
            text = message,
            data = toolErrorData("invalid_arguments", message),
        )

    private fun parseArgv(element: JsonElement): List<String>? {
        val array = element as? JsonArray ?: return null
        if (array.isEmpty()) {
            return null
        }

        val parts =
            array.map { item ->
                val primitive = item as? JsonPrimitive ?: return null
                if (!primitive.isString) {
                    return null
                }
                primitive.content
            }

        return parts.takeIf { it.firstOrNull()?.isNotBlank() == true }
    }

    internal companion object {
        fun forTesting(
            options: BashToolOptions = BashToolOptions(),
            executor: BashCommandExecutor,
        ): BashToolFactory = BashToolFactory(options, executor)
    }
}

internal data class BashExecutionRequest(
    val argv: List<String>,
    val command: String? = null,
    val workingDirectory: String? = null,
    val environment: Map<String, String> = emptyMap(),
    val timeoutMillis: Long,
    val maxOutputChars: Int,
)

internal data class BashExecutionResult(
    val exitCode: Int?,
    val stdout: String,
    val stderr: String,
    val timedOut: Boolean = false,
)

internal fun interface BashCommandExecutor {
    suspend fun run(request: BashExecutionRequest): BashExecutionResult
}

private class ProcessBuilderBashCommandExecutor : BashCommandExecutor {
    override suspend fun run(request: BashExecutionRequest): BashExecutionResult =
        withContext(Dispatchers.IO) {
            val process =
                ProcessBuilder(request.argv)
                    .apply {
                        request.workingDirectory?.let { directory(File(it)) }
                        environment().putAll(request.environment)
                    }.start()

            supervisorScope {
                val stdoutDeferred =
                    async(Dispatchers.IO) {
                        process.inputStream.bufferedReader().use { it.readText() }
                    }
                val stderrDeferred =
                    async(Dispatchers.IO) {
                        process.errorStream.bufferedReader().use { it.readText() }
                    }

                val finished = process.waitFor(request.timeoutMillis, TimeUnit.MILLISECONDS)
                if (!finished) {
                    process.destroy()
                    if (process.isAlive) {
                        process.destroyForcibly()
                    }
                    process.waitFor()
                }

                BashExecutionResult(
                    exitCode = if (finished) process.exitValue() else null,
                    stdout = stdoutDeferred.await(),
                    stderr = stderrDeferred.await(),
                    timedOut = !finished,
                )
            }
        }
}

private fun List<String>.toJsonArray() =
    buildJsonArray {
        this@toJsonArray.forEach { add(JsonPrimitive(it)) }
    }

private fun String.limitTo(maxChars: Int): String =
    when {
        maxChars <= 0 -> ""
        length <= maxChars -> this
        else -> take(maxChars)
    }
