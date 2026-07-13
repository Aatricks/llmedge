package io.aatricks.llmedge.huggingface

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class HFFileSelectionSupportTest {
    private val extensions = listOf(".gguf", ".bin", ".safetensors", ".ckpt", ".pt")

    private fun file(path: String, size: Long) =
        HFModelTree.HFModelFile(type = "file", size = size, path = path)

    private val tree =
        listOf(
            HFModelTree.HFModelFile(type = "directory", size = 0, path = "text_encoders"),
            file("sd3.5_large_fp8_scaled.safetensors", 14_900_000_000L),
            file("text_encoders/clip_l.safetensors", 250_000_000L),
            file("text_encoders/clip_g.safetensors", 1_390_000_000L),
            file("README.md", 1_000L),
        )

    @Test
    fun `selectRepoFile matches subdirectory path exactly`() {
        val selected =
            HFFileSelectionSupport.selectRepoFile(tree, "text_encoders/clip_l.safetensors", extensions)
        assertEquals("text_encoders/clip_l.safetensors", selected?.path)
    }

    @Test
    fun `selectRepoFile matches by path suffix`() {
        val selected = HFFileSelectionSupport.selectRepoFile(tree, "clip_g.safetensors", extensions)
        assertEquals("text_encoders/clip_g.safetensors", selected?.path)
    }

    @Test
    fun `selectRepoFile returns null instead of falling back when explicit filename is missing`() {
        val selected = HFFileSelectionSupport.selectRepoFile(tree, "does_not_exist.safetensors", extensions)
        assertNull(selected)
    }

    @Test
    fun `selectRepoFile without filename picks largest allowed file`() {
        val selected = HFFileSelectionSupport.selectRepoFile(tree, null, extensions)
        assertEquals("sd3.5_large_fp8_scaled.safetensors", selected?.path)
    }
}
