package io.aatricks.llmedge.vision

import io.mockk.mockk
import org.junit.Assert.assertEquals
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

}
