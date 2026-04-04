package io.aatricks.llmedge.vision

import io.aatricks.llmedge.runtime.GGUFReader
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.After
import org.junit.Before
import org.junit.Test

class VisionPromptSupportTest {
    @Before
    fun setUp() {
        GGUFReader.overrideNativeBridgeForTests { _ ->
            object : GGUFReader.NativeBridge {
                override fun getGGUFContextNativeHandle(modelPath: String): Long = 1L
                override fun getContextSize(nativeHandle: Long): Long = 4096L
                override fun getChatTemplate(nativeHandle: Long): String = ""
                override fun getArchitecture(nativeHandle: Long): String = "llava"
                override fun getParameterCount(nativeHandle: Long): String = "7B"
                override fun getModelName(nativeHandle: Long): String = "LLaVA Test Model"
                override fun getFileType(nativeHandle: Long): Int = -1
                override fun getDominantTensorType(nativeHandle: Long): Int = -1
                override fun releaseGGUFContext(nativeHandle: Long) = Unit
            }
        }
    }

    @After
    fun tearDown() {
        GGUFReader.resetNativeBridgeForTests()
    }

    @Test
    fun `appearsVisionCapable detects known multimodal model names`() {
        assertTrue(VisionPromptSupport.appearsVisionCapable("/tmp/llava-phi.gguf"))
        assertTrue(VisionPromptSupport.appearsVisionCapable("/tmp/multimodal-model.gguf"))
        assertFalse(VisionPromptSupport.appearsVisionCapable("/tmp/plain-llm.gguf"))
    }

    @Test
    fun `projector helpers require readable mmproj file`() {
        val mmproj = File.createTempFile("vision-mmproj", ".gguf").apply {
            writeText("mmproj")
            deleteOnExit()
        }

        assertTrue(VisionPromptSupport.hasProjectorSupport(mmproj.absolutePath))
        assertTrue(
            VisionPromptSupport.isReadyForMultimodalInference(
                "/tmp/llava-phi.gguf",
                mmproj.absolutePath,
            ),
        )
        assertFalse(VisionPromptSupport.hasProjectorSupport(null))
        assertFalse(VisionPromptSupport.hasProjectorSupport("/tmp/missing-mmproj.gguf"))
        assertFalse(
            VisionPromptSupport.isReadyForMultimodalInference(
                "/tmp/plain-llm.gguf",
                mmproj.absolutePath,
            ),
        )
    }

    @Test
    fun `unsupportedReason explains missing capability`() {
        val noVisionModelMessage =
            VisionPromptSupport.unsupportedReason("/tmp/plain-llm.gguf", "/tmp/mmproj.gguf")
        val missingProjectorMessage =
            VisionPromptSupport.unsupportedReason("/tmp/llava-phi.gguf", null)

        assertTrue(noVisionModelMessage.contains("does not appear to be a vision-capable GGUF"))
        assertTrue(missingProjectorMessage.contains("requires a readable mmproj/projector file"))
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

    @Test
    fun `appearsVisionCapable prefers GGUF metadata for local files`() {
        val model = File.createTempFile("plain-llm", ".gguf").apply {
            outputStream().use { output ->
                output.write(byteArrayOf('G'.code.toByte(), 'G'.code.toByte(), 'U'.code.toByte(), 'F'.code.toByte()))
                output.write(byteArrayOf(0, 0, 0, 0))
            }
            deleteOnExit()
        }

        assertTrue(VisionPromptSupport.appearsVisionCapable(model.absolutePath))
    }
}
