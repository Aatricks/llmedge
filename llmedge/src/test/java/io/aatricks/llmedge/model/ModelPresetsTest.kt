package io.aatricks.llmedge.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelPresetsTest {
    @Test
    fun `bitnet preset points at the ik_llama IQ2_BN gguf`() {
        val spec = ModelPresets.bitnet as ModelSpec.HuggingFace
        assertEquals("tdh111/bitnet-b1.58-2B-4T-GGUF", spec.repoId)
        assertEquals("bitnet1582b4t-iq2_bn.gguf", spec.filename)
        assertEquals(ModelArtifactKind.GGUF_MODEL, spec.hints.artifactKind)
        assertTrue(ModelCapability.TEXT in spec.hints.capabilities)
    }

    @Test
    fun `bitnet preset carries its own chat template so it works without caller config`() {
        val template = ModelPresets.bitnet.hints.chatTemplate
        assertTrue("BitNet preset must carry a non-empty chat template", !template.isNullOrBlank())
        // Canonical BitNet b1.58 template markers (microsoft/bitnet-b1.58-2B-4T tokenizer_config.json).
        assertTrue(template!!.contains("<|eot_id|>"))
        assertTrue(template.contains("Assistant: "))
        assertTrue(template.contains("add_generation_prompt"))
    }

    @Test
    fun `smolVlm2 preset exposes base model and projector with matching repo`() {
        val model = ModelPresets.smolVlm2.model as ModelSpec.HuggingFace
        val projector = ModelPresets.smolVlm2.projector as ModelSpec.HuggingFace

        assertEquals("ggml-org/SmolVLM2-256M-Video-Instruct-GGUF", model.repoId)
        assertEquals("SmolVLM2-256M-Video-Instruct-Q8_0.gguf", model.filename)
        assertEquals(ModelArtifactKind.GGUF_MODEL, model.hints.artifactKind)
        assertTrue(ModelCapability.VISION in model.hints.capabilities)

        assertEquals("ggml-org/SmolVLM2-256M-Video-Instruct-GGUF", projector.repoId)
        assertEquals("mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf", projector.filename)
        assertEquals(ModelArtifactKind.PROJECTOR, projector.hints.artifactKind)
        assertTrue(ModelCapability.PROJECTOR in projector.hints.capabilities)
    }

    @Test
    fun `default model hints leave chat template unset`() {
        assertEquals(null, ModelHints().chatTemplate)
    }
}
