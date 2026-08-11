package io.aatricks.llmedge.huggingface

import org.junit.Assert.assertEquals
import org.junit.Test

class HuggingFaceHubSanitizeTest {
    @Test
    fun `sanitize replaces slashes with underscores`() {
        val input = "unsloth/Qwen3-0.6B-GGUF"
        val sanitized = HuggingFaceHub.sanitize(input)
        assertEquals("unsloth_Qwen3-0.6B-GGUF", sanitized)
    }

    @Test
    fun `relativeCachePath keeps variant directories distinct`() {
        val b16 = HFFileSelectionSupport.relativeCachePath("minit2i-b-16/transformer/diffusion_pytorch_model.safetensors")
        val l16 = HFFileSelectionSupport.relativeCachePath("minit2i-l-16/transformer/diffusion_pytorch_model.safetensors")
        assertEquals("minit2i-b-16/transformer/diffusion_pytorch_model.safetensors", b16)
        assertEquals("minit2i-l-16/transformer/diffusion_pytorch_model.safetensors", l16)
    }

    @Test
    fun `relativeCachePath drops traversal segments`() {
        assertEquals("etc/passwd", HFFileSelectionSupport.relativeCachePath("../../etc/passwd"))
        assertEquals("model.safetensors", HFFileSelectionSupport.relativeCachePath("model.safetensors"))
    }
}
