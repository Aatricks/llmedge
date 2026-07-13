package io.aatricks.llmedge.image

import io.aatricks.llmedge.model.ModelSpec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class MiniT2ITest {
    @Test
    fun `image request uses MiniT2I transformer and FLAN T5 encoder`() {
        val request = MiniT2I.imageRequest(prompt = "a cat", seed = 42L)

        val model = request.model as ModelSpec.HuggingFace
        assertEquals("MiniT2I/MiniT2I", model.repoId)
        assertEquals("minit2i-b-16/transformer/diffusion_pytorch_model.safetensors", model.filename)

        val textEncoder = request.textEncoder as ModelSpec.HuggingFace
        assertEquals("google/flan-t5-large", textEncoder.repoId)
        assertEquals("model.safetensors", textEncoder.filename)

        assertSame(MiniT2I.diffusionModel, request.model)
        assertSame(MiniT2I.textEncoder, request.textEncoder)
        assertNull(request.vae)
        assertTrue(request.diffusionModelOnly)
        assertFalse(request.splitDiffusionModel)
        assertEquals(100, request.steps)
        assertEquals(6.0f, request.cfgScale)
        assertEquals(42L, request.seed)
    }
}
