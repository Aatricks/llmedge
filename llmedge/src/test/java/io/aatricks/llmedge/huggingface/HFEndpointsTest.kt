package io.aatricks.llmedge.huggingface

import org.junit.Assert.assertEquals
import org.junit.Test

class HFEndpointsTest {
    @Test
    fun `listModelsEndpoint returns base`() {
        assertEquals("https://huggingface.co/api/models", HFEndpoints.listModelsEndpoint())
    }

    @Test
    fun `modelTreeEndpoint constructs tree url`() {
        val url = HFEndpoints.modelTreeEndpoint("user/repo", "main")
        assertEquals("https://huggingface.co/api/models/user/repo/tree/main", url)
    }

    @Test
    fun `modelSpecsEndpoint builds expected path`() {
        val url = HFEndpoints.modelSpecsEndpoint("unsloth/Qwen3-0.6B-GGUF")
        assertEquals("https://huggingface.co/api/models/unsloth/Qwen3-0.6B-GGUF", url)
    }

    @Test
    fun `fileDownloadEndpoint constructs file url`() {
        val url = HFEndpoints.fileDownloadEndpoint("unsloth/Qwen3-0.6B-GGUF", "main", "weights.gguf")
        assertEquals("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/weights.gguf", url)
    }
}
