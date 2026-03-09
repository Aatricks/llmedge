package io.aatricks.llmedge

import io.aatricks.llmedge.runtime.GGUFReader
import io.mockk.*
import org.junit.Assert.assertEquals
import org.junit.Test
import io.aatricks.llmedge.text.runtime.SmolLM

class SmolLMResolveChatTemplateTest {
    @Test
    fun `explicit template takes precedence`() {
        val smol = SmolLM()
        val reader = mockk<GGUFReader>(relaxed = true)
        every { reader.getChatTemplate() } returns "reader-template"

        val method = SmolLM::class.java.getDeclaredMethod("resolveChatTemplate", String::class.java, GGUFReader::class.java)
        method.isAccessible = true
        val result = method.invoke(smol, "explicit", reader) as String
        assertEquals("explicit", result)
    }

    @Test
    fun `reader template used when explicit is null`() {
        val smol = SmolLM()
        val reader = mockk<GGUFReader>(relaxed = true)
        every { reader.getChatTemplate() } returns "reader-template"

        val method = SmolLM::class.java.getDeclaredMethod("resolveChatTemplate", String::class.java, GGUFReader::class.java)
        method.isAccessible = true
        val result = method.invoke(smol, null, reader) as String
        assertEquals("reader-template", result)
    }

    @Test
    fun `default chat template used when both are null`() {
        val smol = SmolLM()
        val reader = mockk<GGUFReader>(relaxed = true)
        every { reader.getChatTemplate() } returns null

        val method = SmolLM::class.java.getDeclaredMethod("resolveChatTemplate", String::class.java, GGUFReader::class.java)
        method.isAccessible = true
        val result = method.invoke(smol, null, reader) as String
        // Default template should not be null and should contain 'assistant'
        assertEquals("Default template should contain 'assistant'", true, result.contains("assistant"))
    }
}
