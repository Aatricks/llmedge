package io.aatricks.llmedge.tools

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class JsonExtractorTest {
    @Test
    fun `extractToolCallJson reads fenced markdown json`() {
        val json =
            JsonExtractor.extractToolCallJson(
                """
                Sure, here you go:
                ```json
                {"tool_name":"weather","arguments":{"city":"Paris"}}
                ```
                """.trimIndent(),
            )

        assertEquals("weather", json?.getString("tool_name"))
        assertEquals("Paris", json?.getJSONObject("arguments")?.getString("city"))
    }

    @Test
    fun `extractToolCallJson reads inline json`() {
        val json = JsonExtractor.extractToolCallJson("Call this: {\"tool_name\":\"search\",\"arguments\":{}} now")

        assertEquals("search", json?.getString("tool_name"))
    }

    @Test
    fun `extractToolCallJson returns null when tool name missing`() {
        assertNull(JsonExtractor.extractToolCallJson("{\"foo\":\"bar\"}"))
    }
}