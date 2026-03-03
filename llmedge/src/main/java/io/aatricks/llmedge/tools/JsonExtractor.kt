package io.aatricks.llmedge.tools

import org.json.JSONException
import org.json.JSONObject

/**
 * Utility to reliably extract and parse JSON from the potentially noisy text output of a local LLM.
 */
object JsonExtractor {

    /**
     * Attempts to find and parse a JSON object containing a "tool_name" key from the given text.
     * Uses multiple fallback strategies to handle various common LLM output formats.
     */
    fun extractToolCallJson(text: String): JSONObject? {
        // 1. Try to find a markdown json block first: ```json ... ``` or ``` ... ```
        val markdownRegex = """```(?:json)?\s*(\{.*\})\s*```""".toRegex(RegexOption.DOT_MATCHES_ALL)
        val markdownMatch = markdownRegex.find(text)
        if (markdownMatch != null) {
            try {
                val obj = JSONObject(markdownMatch.groupValues[1])
                if (obj.has("tool_name")) {
                    return obj
                }
            } catch (e: JSONException) {
                // Ignore and fall through
            }
        }

        // 2. Fallback: Find the first JSON-like object string specifically containing "tool_name"
        val jsonLikeRegex = """\{[^\{\}"]*"tool_name"[^\{\}"]*\}""".toRegex(RegexOption.DOT_MATCHES_ALL)
        val jsonMatch = jsonLikeRegex.find(text)
        if (jsonMatch != null) {
            try {
                val obj = JSONObject(jsonMatch.value)
                if (obj.has("tool_name")) {
                    return obj
                }
            } catch (e: JSONException) {
                // Ignore and fall through
            }
        }
        
        // 3. Final fallback: try parsing any JSON-like block {} found in the text
        val generalJsonRegex = """\{.*\}""".toRegex(RegexOption.DOT_MATCHES_ALL)
        val matches = generalJsonRegex.findAll(text)
        for (match in matches) {
            try {
                val obj = JSONObject(match.value)
                if (obj.has("tool_name")) {
                    return obj
                }
            } catch (e: JSONException) {
                continue
            }
        }

        return null
    }
}
