package io.aatricks.llmedge.vision

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VisionPromptSupportTest {
    @Test
    fun `appearsVisionCapable detects known multimodal model names`() {
        assertTrue(VisionPromptSupport.appearsVisionCapable("/tmp/llava-phi.gguf"))
        assertTrue(VisionPromptSupport.appearsVisionCapable("/tmp/multimodal-model.gguf"))
        assertFalse(VisionPromptSupport.appearsVisionCapable("/tmp/plain-llm.gguf"))
    }

    @Test
    fun `formatVisionPrompt wraps plain prompts`() {
        val formatted = VisionPromptSupport.formatVisionPrompt("Describe this", File("/tmp/image.jpg"))

        assertTrue(formatted.contains("[Image: /tmp/image.jpg]"))
        assertTrue(formatted.contains("User: Describe this"))
    }

    @Test
    fun `formatVisionPrompt preserves system style prompts`() {
        val prompt = "SYSTEM: keep me\nEXAMPLES:\nUser: hi\nAssistant: hello"

        assertEquals(prompt, VisionPromptSupport.formatVisionPrompt(prompt, File("/tmp/image.jpg")))
    }

    @Test
    fun `estimateTokens keeps a minimum of one`() {
        assertEquals(1, VisionPromptSupport.estimateTokens(""))
        assertEquals(25, VisionPromptSupport.estimateTokens("a".repeat(100)))
    }
}