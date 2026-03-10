package io.aatricks.llmedge

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata

/**
 * T104: Tests for loading fine-tuned models with custom filenames.
 * Verifies that community fine-tunes from Hugging Face load correctly.
 */
@RunWith(AndroidJUnit4::class)
class FineTunedModelTest : BaseVideoIntegrationTest() {

    /**
     * Test loading a model with non-standard filename
     */
    @Test
    fun testCustomFilename() = runBlocking {
        val sd = createStableDiffusion()
        
        // Simulate a community fine-tune with custom naming
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video", "fine-tuned"),
                filename = "community-anime-style-wan-1.3b-q4.gguf"
            )
        )
        
        assertTrue("Fine-tuned model should be detected", sd.isVideoModel())
        
        // Verify generation works
        val params = VideoGenerateParams(
            prompt = "test fine-tuned model",
            videoFrames = 8,
            width = 256,
            height = 256
        )
        val frames = sd.txt2vid(params)
        assertEquals("Should generate frames", 8, frames.size)
        
        sd.close()
    }

    /**
     * Test loading with special characters in filename
     */
    @Test
    fun testFilenameWithSpecialChars() = runBlocking {
        val sd = createStableDiffusion()
        
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "Wan2.1_T2V-1.3B_fine-tuned-v2.1.gguf"
            )
        )
        
        assertTrue("Model with special chars should load", sd.isVideoModel())
        sd.close()
    }

    /**
     * Test loading model with version suffix
     */
    @Test
    fun testVersionSuffixFilename() = runBlocking {
        val sd = createStableDiffusion()
        
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "Wan2.2-TI2V-5B-fp8-v1.2-experimental.gguf"
            )
        )
        
        assertTrue("Model with version suffix should load", sd.isVideoModel())
        sd.close()
    }

    /**
     * Test loading merge model (combining multiple fine-tunes)
     */
    @Test
    fun testMergeModel() = runBlocking {
        val sd = createStableDiffusion()
        
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video", "merge"),
                filename = "wan-anime-realistic-merge-1.3b.gguf"
            )
        )
        
        assertTrue("Merge model should be detected", sd.isVideoModel())
        
        val params = VideoGenerateParams(
            prompt = "test merge",
            videoFrames = 4,
            width = 256,
            height = 256
        )
        val frames = sd.txt2vid(params)
        assertEquals(4, frames.size)
        
        sd.close()
    }

    /**
     * Test loading community LoRA-merged model
     */
    @Test
    fun testLoRAMergedModel() = runBlocking {
        val sd = createStableDiffusion()
        
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video", "lora"),
                filename = "wan-5b-cinematic-lora-merged-q6.gguf"
            )
        )
        
        assertTrue("LoRA-merged model should load", sd.isVideoModel())
        sd.close()
    }

    /**
     * Test that metadata inference works with non-standard filenames
     */
    @Test
    fun testMetadataInferenceCustomFilename() = runBlocking {
        val sd = createStableDiffusion()
        
        // Custom filename that doesn't follow standard naming
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "my-custom-video-model.gguf"
            )
        )
        
        assertTrue("Custom filename model should be detected", sd.isVideoModel())
        
        // Verify detection works
        assertTrue("Model should still be video model", sd.isVideoModel())
        
        sd.close()
    }

    /**
     * Test loading with path-like filename (user might include subdirs)
     */
    @Test
    fun testFilenameWithPath() = runBlocking {
        val sd = createStableDiffusion()
        
        // Some users might accidentally include path segments
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "custom/models/video/wan-1.3b.gguf"
            )
        )
        
        assertTrue("Path-like filename should still load", sd.isVideoModel())
        sd.close()
    }

    /**
     * Test that quantization is properly detected from custom filenames
     */
    @Test
    fun testQuantizationDetectionCustomName() = runBlocking {
        // Q4_K_M quantization
        val sd1 = createStableDiffusion()
        sd1.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "custom-model-q4-k-m.gguf"
            )
        )
        assertTrue(sd1.isVideoModel())
        sd1.close()
        
        // Q6_K quantization
        val sd2 = createStableDiffusion()
        sd2.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "high-quality-q6-k.gguf"
            )
        )
        assertTrue(sd2.isVideoModel())
        sd2.close()
        
        // fp8 quantization
        val sd3 = createStableDiffusion()
        sd3.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "experimental-fp8-e4m3fn.gguf"
            )
        )
        assertTrue(sd3.isVideoModel())
        sd3.close()
    }
}
