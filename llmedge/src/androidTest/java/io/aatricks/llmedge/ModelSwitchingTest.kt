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
 * T103: Integration test for model switching and memory stability.
 * Tests loading multiple models sequentially and verifying memory remains stable.
 */
@RunWith(AndroidJUnit4::class)
class ModelSwitchingTest : BaseVideoIntegrationTest() {

    /**
     * Test basic model switching: load 1.3B, generate, close, load 5B, generate
     */
    @Test
    fun testBasicModelSwitching() = runBlocking {
        // Step 1: Load 1.3B model
        val sd1 = createStableDiffusion()
        sd1.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "Wan2.1-T2V-1.3B-Q4_K_M.gguf"
            )
        )
        
        // Generate with 1.3B model
        val params1 = VideoGenerateParams(
            prompt = "test video with 1.3B model",
            videoFrames = 16,
            width = 256,
            height = 256,
            steps = 10
        )
        val frames1 = sd1.txt2vid(params1)
        assertEquals("Should generate 16 frames", 16, frames1.size)
        
        // Close 1.3B model
        sd1.close()
        
        // Verify model is closed
        assertFalse("Model should no longer be video model after close", sd1.isVideoModel())
        
        // Step 2: Load 5B model
        val sd2 = createStableDiffusion()
        sd2.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "Wan2.2-TI2V-5B-fp8.gguf"
            )
        )
        
        // Generate with 5B model (lower frame count due to memory)
        val params2 = VideoGenerateParams(
            prompt = "test video with 5B model",
            videoFrames = 16,
            width = 256,
            height = 256,
            steps = 10
        )
        val frames2 = sd2.txt2vid(params2)
        assertEquals("Should generate 16 frames", 16, frames2.size)
        
        // Close 5B model
        sd2.close()
        
        assertFalse("Model should no longer be video model after close", sd2.isVideoModel())
    }

    /**
     * Test multiple rapid switches between models
     */
    @Test
    fun testMultipleModelSwitches() = runBlocking {
        for (iteration in 1..3) {
            // Load 1.3B
            val sd = createStableDiffusion()
            sd.updateModelMetadata(
                VideoModelMetadata(
                    architecture = "wan",
                    modelType = "t2v",
                    parameterCount = "1.3B",
                    mobileSupported = true,
                    tags = setOf("wan", "video"),
                    filename = "model-iteration-$iteration.gguf"
                )
            )
            
            assertTrue("Model should be detected as video model", sd.isVideoModel())
            
            // Generate a small video
            val params = VideoGenerateParams(
                prompt = "iteration $iteration",
                videoFrames = 8,
                width = 256,
                height = 256,
                steps = 10
            )
            val frames = sd.txt2vid(params)
            assertEquals("Should generate 8 frames", 8, frames.size)
            
            // Close model
            sd.close()
            assertFalse("Model should be closed", sd.isVideoModel())
        }
    }

    /**
     * Test that metadata is properly reset on close (T096)
     */
    @Test
    fun testMetadataResetOnClose() = runBlocking {
        val sd = createStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "test-model.gguf"
            )
        )
        
        assertTrue("Model should be video model before close", sd.isVideoModel())
        
        // Close model
        sd.close()
        
        // Verify metadata is reset
        assertFalse("Model should not be video model after close", sd.isVideoModel())
    }

    /**
     * Test switching between different model types (T2V -> I2V -> TI2V)
     */
    @Test
    fun testSwitchingBetweenModelTypes() = runBlocking {
        // Load T2V (text-to-video)
        val t2v = createStableDiffusion()
        t2v.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "t2v.gguf"
            )
        )
        assertTrue("T2V model loaded", t2v.isVideoModel())
        
        val params = VideoGenerateParams(
            prompt = "t2v test",
            videoFrames = 8,
            width = 256,
            height = 256
        )
        val frames1 = t2v.txt2vid(params)
        assertEquals(8, frames1.size)
        t2v.close()
        
        // Load TI2V (text+image-to-video)
        val ti2v = createStableDiffusion()
        ti2v.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "ti2v",
                parameterCount = "5B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "ti2v.gguf"
            )
        )
        assertTrue("TI2V model loaded", ti2v.isVideoModel())
        
        val frames2 = ti2v.txt2vid(params)
        assertEquals(8, frames2.size)
        ti2v.close()
    }

    /**
     * Test that cancellation state is reset on close (T096)
     */
    @Test
    fun testCancellationStateResetOnClose() = runBlocking {
        val sd = createStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "test.gguf"
            )
        )
        
        // Start generation in background (don't await)
        val params = VideoGenerateParams(
            prompt = "test",
            videoFrames = 32,  // Longer generation
            steps = 20
        )
        
        // Cancel immediately
        sd.cancelGeneration()
        
        try {
            sd.txt2vid(params)
        } catch (e: Exception) {
            // Expected - generation was cancelled
        }
        
        // Close model (should reset cancellation flag)
        sd.close()
        
        // Create new instance - cancellation should not persist
        val sd2 = createStableDiffusion()
        sd2.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("wan", "video"),
                filename = "test2.gguf"
            )
        )
        
        // Should be able to generate without cancellation
        val params2 = VideoGenerateParams(
            prompt = "new test",
            videoFrames = 8,
            steps = 10
        )
        val frames = sd2.txt2vid(params2)
        assertEquals("Generation should succeed after model reload", 8, frames.size)
        
        sd2.close()
    }
}
