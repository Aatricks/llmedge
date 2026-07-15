package io.aatricks.llmedge.image

import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelSpec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ChromaRadianceTest {
    @Test
    fun testImageRequestDefaults() {
        val request = ChromaRadiance.imageRequest(
            prompt = "A majestic dragon"
        )

        assertEquals("A majestic dragon", request.prompt)
        assertEquals("", request.negative)
        assertEquals(512, request.width)
        assertEquals(512, request.height)
        assertEquals(20, request.steps)
        assertEquals(4.0f, request.cfgScale)
        assertEquals(-1L, request.seed)
        assertTrue(request.flashAttention)

        // Model Specs
        val ditSpec = request.model as ModelSpec.HuggingFace
        assertEquals("silveroxides/Chroma1-Radiance-GGUF", ditSpec.repoId)
        assertEquals("Chroma1-Radiance-v0.3/Chroma1-Radiance-v0.3-Q4_K_S.gguf", ditSpec.filename)
        assertEquals(ModelArtifactKind.DIFFUSION_MODEL, ditSpec.hints.artifactKind)
        assertTrue(ditSpec.hints.capabilities.contains(ModelCapability.IMAGE))

        val t5Spec = request.t5xxl as ModelSpec.HuggingFace
        assertEquals("city96/t5-v1_1-xxl-encoder-gguf", t5Spec.repoId)
        assertEquals("t5-v1_1-xxl-encoder-Q3_K_S.gguf", t5Spec.filename)
        assertEquals(ModelArtifactKind.TEXT_ENCODER, t5Spec.hints.artifactKind)
        assertTrue(t5Spec.hints.capabilities.contains(ModelCapability.TEXT))
        assertTrue(t5Spec.hints.capabilities.contains(ModelCapability.IMAGE))

        assertNull(request.vae)
        assertNull(request.clipL)
        assertNull(request.clipG)
        assertNull(request.textEncoder)
        assertTrue(request.splitDiffusionModel)
        assertEquals(true, request.sequential)
    }

    @Test
    fun testMobileImageRequestDefaults() {
        val request = ChromaRadiance.mobileImageRequest(prompt = "A majestic dragon")

        val ditSpec = request.model as ModelSpec.HuggingFace
        assertEquals("silveroxides/Chroma1-HD-GGUF", ditSpec.repoId)
        assertEquals("Chroma1-HD-Q3_K_S.gguf", ditSpec.filename)
        assertEquals(ChromaRadiance.t5xxl, request.t5xxl)
        assertEquals(ChromaRadiance.fluxVae, request.vae)
        assertTrue(request.splitDiffusionModel)
        assertEquals(true, request.sequential)
    }

    @Test
    fun testImageRequestOverrides() {
        val request = ChromaRadiance.imageRequest(
            prompt = "A majestic dragon",
            negative = "low quality",
            width = 256,
            height = 256,
            steps = 30,
            cfgScale = 5.0f,
            seed = 12345L,
            flashAttention = false,
            sequential = false
        )

        assertEquals("low quality", request.negative)
        assertEquals(256, request.width)
        assertEquals(256, request.height)
        assertEquals(30, request.steps)
        assertEquals(5.0f, request.cfgScale)
        assertEquals(12345L, request.seed)
        assertTrue(!request.flashAttention)
        assertEquals(false, request.sequential)
    }

    @Test
    fun testChromaVaeConfiguration() {
        val mobileRequest = ChromaRadiance.mobileImageRequest(prompt = "A majestic dragon")
        val vaeSpec = mobileRequest.vae as? ModelSpec.HuggingFace
        org.junit.Assert.assertNotNull("Mobile image request VAE should not be null", vaeSpec)
        assertEquals("lodestones/Chroma", vaeSpec?.repoId)
        assertEquals("ae.safetensors", vaeSpec?.filename)

        val radianceRequest = ChromaRadiance.imageRequest(prompt = "A majestic dragon")
        assertNull("Radiance image request VAE should be null", radianceRequest.vae)
    }
}
