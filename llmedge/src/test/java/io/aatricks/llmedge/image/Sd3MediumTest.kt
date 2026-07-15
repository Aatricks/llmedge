package io.aatricks.llmedge.image

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class Sd3MediumTest {
    @Test
    fun testImageRequest() {
        val request = Sd3Medium.imageRequest(
            prompt = "A cute cat",
            negative = "ugly",
            width = 256,
            height = 256,
            steps = 28,
            cfgScale = 4.5f,
            seed = 42L,
            flashAttention = true
        )

        assertEquals("A cute cat", request.prompt)
        assertEquals("ugly", request.negative)
        assertEquals(256, request.width)
        assertEquals(256, request.height)
        assertEquals(28, request.steps)
        assertEquals(4.5f, request.cfgScale)
        assertEquals(42L, request.seed)
        assertTrue(request.flashAttention)
        assertEquals(Sd3Medium.diffusionModel, request.model)
        assertEquals(Sd3Medium.vae, request.vae)
        assertEquals(Sd3Medium.clipL, request.clipL)
        assertEquals(Sd3Medium.clipG, request.clipG)
        assertEquals(Sd3Medium.t5xxl, request.t5xxl)
        assertNull(request.textEncoder)
        assertTrue(request.splitDiffusionModel)
        assertNull(request.sequential)
    }

    @Test
    fun testAllInOneImageRequest() {
        val request = Sd3Medium.allInOneImageRequest(
            prompt = "A cute cat",
            negative = "ugly",
            width = 256,
            height = 256,
            steps = 28,
            cfgScale = 4.5f,
            seed = 42L,
            flashAttention = true
        )

        assertEquals("A cute cat", request.prompt)
        assertEquals("ugly", request.negative)
        assertEquals(256, request.width)
        assertEquals(256, request.height)
        assertEquals(28, request.steps)
        assertEquals(4.5f, request.cfgScale)
        assertEquals(42L, request.seed)
        assertTrue(request.flashAttention)
        assertEquals(Sd3Medium.allInOneModel, request.model)
        assertNull(request.vae)
        assertNull(request.clipL)
        assertNull(request.clipG)
        assertNull(request.t5xxl)
        assertNull(request.textEncoder)
        assertFalse(request.splitDiffusionModel)
        assertNull(request.sequential)
    }

    @Test
    fun testT5xxlIsQuantizedGguf() {
        val spec = Sd3Medium.t5xxl as io.aatricks.llmedge.model.ModelSpec.HuggingFace
        assertEquals("city96/t5-v1_1-xxl-encoder-gguf", spec.repoId)
        assertEquals("t5-v1_1-xxl-encoder-Q3_K_S.gguf", spec.filename)
    }
}
