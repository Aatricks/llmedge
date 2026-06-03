package io.aatricks.llmedge.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SafetensorsSpecTest {
    @Test
    fun `safetensors factory attaches conversion to hints`() {
        val spec =
            ModelSpec.safetensors(
                repoId = "deepgrove/Bonsai",
                precision = ConversionPrecision.IQ2_BN,
                adapter = ConversionAdapter.BONSAI_QLINEAR,
            ) as ModelSpec.HuggingFace
        assertEquals("deepgrove/Bonsai", spec.repoId)
        val conv = spec.hints.conversion
        assertNotNull(conv)
        assertEquals(ConversionPrecision.IQ2_BN, conv!!.precision)
        assertEquals(ConversionAdapter.BONSAI_QLINEAR, conv.adapter)
        assertTrue(ModelCapability.TEXT in spec.hints.capabilities)
    }

    @Test
    fun `default factories leave conversion null`() {
        val gguf = ModelSpec.huggingFace("a/b", "x.gguf") as ModelSpec.HuggingFace
        assertNull(gguf.hints.conversion)
        assertNull(ModelHints().conversion)
    }

    @Test
    fun `conversion is tagged in the cache key and plain specs are unaffected`() {
        val plain = ModelSpec.huggingFace("a/b", "x.gguf")
        val f16 = ModelSpec.safetensors("a/b", ConversionPrecision.F16)
        val q8 = ModelSpec.safetensors("a/b", ConversionPrecision.Q8_0)

        assertTrue(!plain.cacheKey.contains("convert:"))
        assertTrue(f16.cacheKey.contains("convert:f16:none"))
        assertTrue(f16.cacheKey != q8.cacheKey) // precision differentiates
    }

    @Test
    fun `local safetensors factory uses LocalFile with conversion`() {
        val spec =
            ModelSpec.safetensorsLocal("/models/bonsai", ConversionPrecision.Q4_K_M) as ModelSpec.LocalFile
        assertEquals("/models/bonsai", spec.file.path)
        assertEquals(ConversionPrecision.Q4_K_M, spec.hints.conversion?.precision)
        assertTrue(spec.cacheKey.contains("convert:q4_k_m:none"))
    }
}
