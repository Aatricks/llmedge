package io.aatricks.llmedge.tools

/**
 * Represents a tool that the LLM can invoke.
 *
 * @property name The name of the tool, used by the model to specify which tool to call.
 * @property description A clear description of what the tool does and when to use it.
 * @property parameters A map defining the expected arguments for the tool. The key is the parameter name.
 * @property execute The suspend function to execute when the tool is called. Takes the parsed arguments map and returns a String result.
 */
data class Tool(
    val name: String,
    val description: String,
    val parameters: Map<String, ParameterDescription>,
    val execute: suspend (Map<String, Any>) -> String
)

/**
 * Describes a single parameter for a Tool.
 *
 * @property type The type of the parameter (e.g., "string", "number", "boolean").
 * @property description A clear description of what the parameter represents.
 * @property required Whether the parameter is mandatory.
 */
data class ParameterDescription(
    val type: String,
    val description: String,
    val required: Boolean = true
)
