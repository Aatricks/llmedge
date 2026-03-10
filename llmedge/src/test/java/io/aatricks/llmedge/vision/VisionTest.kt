package io.aatricks.llmedge.vision

import io.mockk.*
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.robolectric.annotation.Config

@Config(sdk = [34])
class VisionTest {

    @Test
    fun `OcrResult data class works correctly`() {
        val result = OcrResult(
            text = "OCR result",
            language = "eng",
            durationMs = 100L,
            engine = "test",
            confidence = 0.85f
        )

        assertEquals("OCR result", result.text)
        assertEquals("eng", result.language)
        assertEquals(100L, result.durationMs)
        assertEquals("test", result.engine)
        assertEquals(0.85f, result.confidence)
    }

    @Test
    fun `VisionResult data class works correctly`() {
        val result = VisionResult(
            text = "Test vision result",
            durationMs = 200L,
            modelId = "test-model",
            tokensIn = 10,
            tokensOut = 20
        )

        assertEquals("Test vision result", result.text)
        assertEquals(200L, result.durationMs)
        assertEquals("test-model", result.modelId)
        assertEquals(10, result.tokensIn)
        assertEquals(20, result.tokensOut)
    }

    @Test
    fun `VisionRuntimeMemory data class works correctly`() {
        val memory = VisionRuntimeMemory(nativeBytes = 1024L, stateBytes = 256L)

        assertEquals(1024L, memory.nativeBytes)
        assertEquals(256L, memory.stateBytes)
    }

    @Test
    fun `VisionRequest carries optional thread configuration`() {
        val request = VisionRequest(
            image = mockk(),
            prompt = "Describe this image",
            model = mockk(),
            projector = mockk(),
            numThreads = 6,
            generationThreads = 2,
        )

        assertEquals("Describe this image", request.prompt)
        assertEquals(6, request.numThreads)
        assertEquals(2, request.generationThreads)
    }

    @Test
    fun `OcrParams data class works correctly`() {
        val params = OcrParams(
            language = "eng",
            pageSegmentationMode = 3,
            engineMode = 3,
            enhance = true
        )

        assertEquals("eng", params.language)
        assertEquals(3, params.pageSegmentationMode)
        assertEquals(3, params.engineMode)
        assertEquals(true, params.enhance)
    }

    @Test
    fun `VisionParams data class works correctly`() {
        val params = VisionParams(
            maxTokens = 256,
            temperature = 0.2f,
            systemPrompt = "Test prompt",
            nBatch = 4
        )

        assertEquals(256, params.maxTokens)
        assertEquals(0.2f, params.temperature)
        assertEquals("Test prompt", params.systemPrompt)
        assertEquals(4, params.nBatch)
    }

    @Test
    fun `VisionMode enum values work correctly`() {
        assertEquals(4, io.aatricks.llmedge.vision.VisionMode.entries.size)
        assertTrue(io.aatricks.llmedge.vision.VisionMode.entries.contains(io.aatricks.llmedge.vision.VisionMode.AUTO_PREFER_VISION))
        assertTrue(io.aatricks.llmedge.vision.VisionMode.entries.contains(io.aatricks.llmedge.vision.VisionMode.FORCE_VISION))
        assertTrue(io.aatricks.llmedge.vision.VisionMode.entries.contains(io.aatricks.llmedge.vision.VisionMode.AUTO_PREFER_OCR))
        assertTrue(io.aatricks.llmedge.vision.VisionMode.entries.contains(io.aatricks.llmedge.vision.VisionMode.FORCE_MLKIT))
    }

    @Test
    fun `ImageUnderstandingResult data class works correctly`() {
        val result = ImageUnderstandingResult(
            text = "Understanding result",
            engine = "test-engine",
            durationMs = 150L,
            confidence = 0.9f
        )

        assertEquals("Understanding result", result.text)
        assertEquals("test-engine", result.engine)
        assertEquals(150L, result.durationMs)
        assertEquals(0.9f, result.confidence)
    }

    @Test
    fun `ImageSource FileSource works correctly`() {
        val file = java.io.File("/test/image.jpg")
        val source = ImageSource.FileSource(file)

        assertEquals(file, source.file)
    }

    @Test
    fun `ImageSource ByteArraySource works correctly`() {
        val bytes = byteArrayOf(1, 2, 3, 4)
        val format = "jpeg"
        val source = ImageSource.ByteArraySource(bytes, format)

        assertEquals(bytes, source.bytes)
        assertEquals(format, source.format)
    }

    @Test
    fun `ImageSource ByteArraySource equals and hashCode work correctly`() {
        val bytes1 = byteArrayOf(1, 2, 3)
        val bytes2 = byteArrayOf(1, 2, 3)
        val bytes3 = byteArrayOf(4, 5, 6)

        val source1 = ImageSource.ByteArraySource(bytes1, "jpeg")
        val source2 = ImageSource.ByteArraySource(bytes2, "jpeg")
        val source3 = ImageSource.ByteArraySource(bytes3, "png")

        assertEquals(source1, source2)
        assertEquals(source1.hashCode(), source2.hashCode())
        assertTrue(source1 != source3)
        assertTrue(source1.hashCode() != source3.hashCode())
    }

    @Test
    fun `SmolLMVisionAdapter checkVisionSupport works correctly`() {
        // Test with mocked SmolLMVisionAdapter to access private method
        val mockContext = mockk<android.content.Context>()
        val mockSmolLM = mockk<io.aatricks.llmedge.text.runtime.SmolLM>()

        val adapter = SmolLMVisionAdapter(mockContext, mockSmolLM)

        // Use reflection to test private method
        val method = SmolLMVisionAdapter::class.java.getDeclaredMethod("checkVisionSupport", String::class.java)
        method.isAccessible = true

        assertTrue(method.invoke(adapter, "/path/to/llava-model.gguf") as Boolean)
        assertTrue(method.invoke(adapter, "/path/to/vision-model.gguf") as Boolean)
        assertTrue(method.invoke(adapter, "/path/to/clip-model.gguf") as Boolean)
        assertTrue(method.invoke(adapter, "/path/to/multimodal-model.gguf") as Boolean)
        assertTrue(method.invoke(adapter, "/path/to/LLAVA-model.gguf") as Boolean) // case insensitive

        assertTrue(!(method.invoke(adapter, "/path/to/regular-model.gguf") as Boolean))
        assertTrue(!(method.invoke(adapter, "/path/to/text-model.gguf") as Boolean))
    }

    @Test
    fun `SmolLMVisionAdapter formatVisionPrompt works correctly`() {
        val mockContext = mockk<android.content.Context>()
        val mockSmolLM = mockk<io.aatricks.llmedge.text.runtime.SmolLM>()

        val adapter = SmolLMVisionAdapter(mockContext, mockSmolLM)

        // Use reflection to test private method
        val method = SmolLMVisionAdapter::class.java.getDeclaredMethod("formatVisionPrompt", String::class.java, java.io.File::class.java)
        method.isAccessible = true

        val imageFile = java.io.File("/test/image.jpg")

        // Test simple prompt formatting
        val result1 = method.invoke(adapter, "Describe this image", imageFile) as String
        assertTrue(result1.contains("Describe this image"))
        assertTrue(result1.contains("[Image: /test/image.jpg]"))

        // Test that system prompts are not double-wrapped
        val systemPrompt = "SYSTEM: You are a helpful assistant.\nEXAMPLES:\nUser: hello\nAssistant: hi"
        val result2 = method.invoke(adapter, systemPrompt, imageFile) as String
        assertEquals(systemPrompt, result2)

        // Test ChatML-formatted prompts used by the example app
        val chatMlPrompt = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nDescribe this image.<|end|>\n<|assistant|>\n"
        val resultChatMl = method.invoke(adapter, chatMlPrompt, imageFile) as String
        assertEquals(chatMlPrompt, resultChatMl)

        // Test OCR markers
        val ocrPrompt = "OCR_TEXT_START: Some text OCR_TEXT_END: Describe this"
        val result3 = method.invoke(adapter, ocrPrompt, imageFile) as String
        assertEquals(ocrPrompt, result3)
    }

    @Test
    fun `SmolLMVisionAdapter estimateTokens works correctly`() {
        val mockContext = mockk<android.content.Context>()
        val mockSmolLM = mockk<io.aatricks.llmedge.text.runtime.SmolLM>()

        val adapter = SmolLMVisionAdapter(mockContext, mockSmolLM)

        // Use reflection to test private method
        val method = SmolLMVisionAdapter::class.java.getDeclaredMethod("estimateTokens", String::class.java)
        method.isAccessible = true

        assertEquals(1, method.invoke(adapter, "")) // minimum 1
        assertEquals(1, method.invoke(adapter, "a")) // 1 char = 1 token
        assertEquals(1, method.invoke(adapter, "abcd")) // 4 chars = 1 token
        assertEquals(1, method.invoke(adapter, "abcde")) // 5 chars = 1 token (5/4 = 1 in integer division)
        assertEquals(25, method.invoke(adapter, "a".repeat(100))) // 100 chars = 25 tokens
    }

    @Test
    fun `SmolLMVisionAdapter getModelId works correctly when no model loaded`() {
        val mockContext = mockk<android.content.Context>()
        val mockSmolLM = mockk<io.aatricks.llmedge.text.runtime.SmolLM>()

        val adapter = SmolLMVisionAdapter(mockContext, mockSmolLM)

        assertEquals("unknown", adapter.getModelId())
    }

    @Test
    fun `SmolLMVisionAdapter hasVisionCapabilities works correctly when no model loaded`() {
        val mockContext = mockk<android.content.Context>()
        val mockSmolLM = mockk<io.aatricks.llmedge.text.runtime.SmolLM>()

        val adapter = SmolLMVisionAdapter(mockContext, mockSmolLM)

        assertTrue(!adapter.hasVisionCapabilities())
    }

    @Test
    fun `SmolLMVisionAdapter analyze uses single-token generation after decoding embeddings`() = runTest {
        val mockContext = mockk<android.content.Context>()
        val mockSmolLM = mockk<io.aatricks.llmedge.text.runtime.SmolLM>()
        val adapter = SmolLMVisionAdapter(mockContext, mockSmolLM)
        val workDir = createTempDir(prefix = "vision-adapter-test")
        val embdFile = java.io.File(workDir, "vision_prepared.bin").apply { writeBytes(byteArrayOf(4, 5, 6)) }
        val metaFile = java.io.File(embdFile.absolutePath + ".meta.json").apply { writeText("{}") }

        val hasVisionSupportField = SmolLMVisionAdapter::class.java.getDeclaredField("hasVisionSupport")
        hasVisionSupportField.isAccessible = true
        hasVisionSupportField.setBoolean(adapter, true)
        val modelPathField = SmolLMVisionAdapter::class.java.getDeclaredField("modelPath")
        modelPathField.isAccessible = true
        modelPathField.set(adapter, "/tmp/llava-model.gguf")
        val mmprojPathField = SmolLMVisionAdapter::class.java.getDeclaredField("mmprojPath")
        mmprojPathField.isAccessible = true
        mmprojPathField.set(adapter, "/tmp/mmproj.gguf")

        mockkObject(ImageUtils)
        try {
            coEvery {
                ImageUtils.imageToFile(mockContext, any(), "vision_input.jpg")
            } returns embdFile
            every {
                mockSmolLM.decodePreparedEmbeddings(
                    embdFile.absolutePath,
                    metaFile.absolutePath,
                    1,
                )
            } returns true
            every { mockSmolLM.getResponse("Describe the image", 256, 1) } returns "vision-result"

            val result =
                adapter.analyze(
                    ImageSource.FileSource(embdFile),
                    "Describe the image",
                    VisionParams(),
                )

            assertEquals("vision-result", result.text)
            verify(exactly = 1) {
                mockSmolLM.decodePreparedEmbeddings(
                    embdFile.absolutePath,
                    metaFile.absolutePath,
                    1,
                )
            }
            verify(exactly = 1) {
                mockSmolLM.getResponse("Describe the image", 256, 1)
            }
        } finally {
            unmockkObject(ImageUtils)
            workDir.deleteRecursively()
        }
    }
}