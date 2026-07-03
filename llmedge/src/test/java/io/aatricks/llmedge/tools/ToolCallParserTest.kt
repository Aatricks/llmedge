package io.aatricks.llmedge.tools

import kotlinx.serialization.json.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ToolCallParserTest {
    @Test
    fun `classify reads fenced markdown json with new tool envelope`() {
        val turn =
            ToolCallParser.classify(
                """
                ```json
                {"tool":"weather","arguments":{"city":"Paris"}}
                ```
                """.trimIndent(),
            )

        assertTrue(turn is ParsedModelTurn.ToolInvocation)
        val call = (turn as ParsedModelTurn.ToolInvocation).call
        assertEquals("weather", call.tool)
        assertEquals("Paris", call.arguments["city"]?.toString()?.trim('"'))
    }

    @Test
    fun `classify accepts legacy tool_name envelope`() {
        val turn = ToolCallParser.classify("""{"tool_name":"search","arguments":{}}""")

        assertTrue(turn is ParsedModelTurn.ToolInvocation)
        assertEquals("search", (turn as ParsedModelTurn.ToolInvocation).call.tool)
        assertEquals(JsonObject(emptyMap()), turn.call.arguments)
    }

    @Test
    fun `classify reports invalid tool payload when arguments are not an object`() {
        val turn = ToolCallParser.classify("""{"tool":"search","arguments":"oops"}""")

        assertTrue(turn is ParsedModelTurn.InvalidToolInvocation)
        assertTrue((turn as ParsedModelTurn.InvalidToolInvocation).reason.contains("arguments"))
    }

    @Test
    fun `classify falls back to final text when no tool call exists`() {
        val turn = ToolCallParser.classify("Just answer normally.")

        assertEquals(ParsedModelTurn.FinalText("Just answer normally."), turn)
    }

    @Test
    fun `classify handles brace-leading invalid JSON as invalid tool invocation`() {
        val turn = ToolCallParser.classify("""{"tool": "weather", "arguments": { invalid }""")
        assertTrue(turn is ParsedModelTurn.InvalidToolInvocation)
    }

    @Test
    fun `classify detects single embedded balanced top level JSON object`() {
        val text = """
            Here is the tool you should call:
            {
              "tool": "get_time",
              "arguments": {}
            }
            Let me know if you need anything else.
        """.trimIndent()
        val turn = ToolCallParser.classify(text)
        assertTrue(turn is ParsedModelTurn.ToolInvocation)
        val call = (turn as ParsedModelTurn.ToolInvocation).call
        assertEquals("get_time", call.tool)
    }
}
