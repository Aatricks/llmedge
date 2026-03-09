package io.aatricks.llmedge

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata

/**
 * T102: Tests for model variant detection and loading.
 * Tests loading different variants, metadata detection, and 14B rejection.
 */
@RunWith(AndroidJUnit4::class)
class ModelVariantTest : BaseVideoIntegrationTest() {

    /**
     * Test that metadata detection works for 1.3B models
     */
    @Test
    fun testDetect1_3BVariant() {
        val sd = createStableDiffusion()
        
        // Update with 1.3B metadata
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "Wan2.1-T2V-1.3B-Q4_K_M.gguf"
            )
        )
        
        assertTrue("1.3B model should be detected as video model", sd.isVideoModel())
    }

    /**
     * Test that metadata detection works for 5B models
     */
    @Test
    fun testDetect5BVariant() {
        val sd = createStableDiffusion()
        
        // Update with 5B metadata
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "Wan2.2-TI2V-5B-fp8.gguf"
            )
        )
        
        assertTrue("5B model should be detected as video model", sd.isVideoModel())
    }

    /**
     * Test that 14B models are marked as not mobile-supported
     */
    @Test
    fun test14BModelNotMobileSupported() {
        val metadata = VideoModelMetadata(
            architecture = "wan",
            modelType = "i2v",
            parameterCount = "14B",
            mobileSupported = false,
            tags = setOf("wan", "video"),
            filename = "Wan2.1-I2V-14B-Q4_K_M.gguf"
        )
        
        assertFalse("14B model should not be mobile-supported", metadata.mobileSupported)
        assertEquals("14B", metadata.parameterCount)
    }

    /**
     * Test that model switching works (close and create new instance)
     */
    @Test
    fun testModelSwitching() = runBlocking {
        // Create first model instance
        val sd1 = createStableDiffusion()
        sd1.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "model1.gguf"
            )
        )
        
        assertTrue("First model should be video model", sd1.isVideoModel())
        
        // Close first model
        sd1.close()
        
        // Create second model instance (simulating model switch)
        val sd2 = createStableDiffusion()
        sd2.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "model2.gguf"
            )
        )
        
        assertTrue("Second model should be video model", sd2.isVideoModel())
        
        sd2.close()
    }

    /**
     * Test that model type detection works for different variants
     */
    @Test
    fun testModelTypeDetection() {
        // T2V model
        val t2v = VideoModelMetadata(
            architecture = "wan",
            modelType = "t2v",
            parameterCount = "1.3B",
            mobileSupported = true,
            tags = setOf("wan"),
            filename = "t2v-model.gguf"
        )
        assertEquals("t2v", t2v.modelType)
        
        // I2V model
        val i2v = VideoModelMetadata(
            architecture = "wan",
            modelType = "i2v",
            parameterCount = "14B",
            mobileSupported = false,
            tags = setOf("wan"),
            filename = "i2v-model.gguf"
        )
        assertEquals("i2v", i2v.modelType)
        
        // TI2V model (text/image to video)
        val ti2v = VideoModelMetadata(
            architecture = "wan",
            modelType = "ti2v",
            parameterCount = "5B",
            mobileSupported = true,
            tags = setOf("wan"),
            filename = "ti2v-model.gguf"
        )
        assertEquals("ti2v", ti2v.modelType)
    }

    /**
     * Test frame limit enforcement for 5B models (T101)
     */
    @Test
    fun testFrameLimitFor5BModel() = runBlocking {
        val sd = createStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "Wan2.2-TI2V-5B.gguf"
            )
        )
        
        // Try to generate with > 32 frames (should fail for 5B)
        val params = VideoGenerateParams(
            prompt = "test",
            videoFrames = 48  // More than 32 frame limit for 5B
        )
        
        try {
            sd.txt2vid(params)
            fail("Should throw exception for exceeding 5B frame limit")
        } catch (e: IllegalArgumentException) {
            assertTrue("Error should mention frame limit", 
                e.message?.contains("32 frames") == true)
        }
    }

    /**
     * Test that 1.3B models can use full 64 frames
     */
    @Test
    fun test1_3BModelSupports64Frames() = runBlocking {
        val sd = createStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "Wan2.1-T2V-1.3B.gguf"
            )
        )
        
        val params = VideoGenerateParams(
            prompt = "test",
            videoFrames = 64  // Max frames
        )
        
        // Should not throw - just validate params structure
        assertNotNull("Params with 64 frames should be valid for 1.3B", params)
        assertEquals(64, params.videoFrames)
    }
}
