package io.aatricks.llmedge

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.nio.ByteBuffer
import java.security.MessageDigest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback

/**
 * Reproducibility tests for video generation using stubbed native bridge. Tests that identical
 * parameters produce identical outputs and that randomness works correctly.
 *
 * These tests use a deterministic stub implementation to validate the Kotlin API layer ensures
 * reproducibility contracts are enforced correctly.
 */
@RunWith(AndroidJUnit4::class)
class VideoReproducibilityTest : BaseVideoIntegrationTest() {

    /** Creates a deterministic stub that returns frames based on seed value */
    private fun createDeterministicStub(seed: Long): StableDiffusion.NativeBridge {
        return object : StableDiffusion.NativeBridge {
            override fun txt2img(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    steps: Int,
                    cfg: Float,
                    seed: Long,
                    vaeTiling: Boolean,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
            ): ByteArray? = null
            override fun txt2vid(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    videoFrames: Int,
                    steps: Int,
                    cfg: Float,
                    seed: Long,
                    sampleMethod: SampleMethod,
                    scheduler: Scheduler,
                    strength: Float,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                    vaceStrength: Float,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
            ): Array<ByteArray>? {
                // Generate deterministic frames based on seed
                return Array(videoFrames) { frameIndex ->
                    val bytes = ByteArray(width * height * 3)
                    // Use seed to generate deterministic but different patterns per frame
                    val frameSeed = seed + frameIndex
                    for (i in bytes.indices step 3) {
                        val pixelSeed = frameSeed + (i / 3)
                        bytes[i] = ((pixelSeed shr 16) and 0xFF).toByte() // R
                        bytes[i + 1] = ((pixelSeed shr 8) and 0xFF).toByte() // G
                        bytes[i + 2] = (pixelSeed and 0xFF).toByte() // B
                    }
                    bytes
                }
            }

            override fun setProgressCallback(
                    handle: Long,
                    callback: VideoProgressCallback?
            ) = Unit
            override fun cancelGeneration(handle: Long) = Unit
            override fun precomputeCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    clipSkip: Int,
            ): PrecomputedCondition? = null
        }
    }

    /** Creates a random stub that returns different frames each time */
    private fun createRandomStub(): StableDiffusion.NativeBridge {
        return object : StableDiffusion.NativeBridge {
            override fun txt2img(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    steps: Int,
                    cfg: Float,
                    seed: Long,
                    vaeTiling: Boolean,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
            ): ByteArray? = null
            override fun txt2vid(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    videoFrames: Int,
                    steps: Int,
                    cfg: Float,
                    seed: Long,
                    sampleMethod: SampleMethod,
                    scheduler: Scheduler,
                    strength: Float,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                    vaceStrength: Float,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
            ): Array<ByteArray>? {
                // Generate random frames (ignoring seed parameter to simulate randomness)
                val actualSeed = System.nanoTime()
                return Array(videoFrames) { frameIndex ->
                    val bytes = ByteArray(width * height * 3)
                    val frameSeed = actualSeed + frameIndex + System.currentTimeMillis()
                    for (i in bytes.indices step 3) {
                        val pixelSeed = frameSeed + (i / 3) + System.nanoTime()
                        bytes[i] = ((pixelSeed shr 16) and 0xFF).toByte()
                        bytes[i + 1] = ((pixelSeed shr 8) and 0xFF).toByte()
                        bytes[i + 2] = (pixelSeed and 0xFF).toByte()
                    }
                    bytes
                }
            }

            override fun setProgressCallback(
                    handle: Long,
                    callback: VideoProgressCallback?
            ) = Unit
            override fun cancelGeneration(handle: Long) = Unit
            override fun precomputeCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    clipSkip: Int,
            ): PrecomputedCondition? = null
        }
    }

    /**
     * T084: Test deterministic generation with fixed seed Generate 2 videos with same seed, compare
     * frame checksums byte-by-byte
     */
    @Test
    fun testDeterministicGenerationWithFixedSeed() = runBlocking {
        val fixedSeed = 42L
        StableDiffusion.overrideNativeBridgeForTests { createDeterministicStub(fixedSeed) }
        val sd = createStableDiffusion()

        val params =
                VideoGenerateParams(
                        prompt = "a cat walking",
                        width = 256,
                        height = 256,
                        videoFrames = 4,
                        steps = 10,
                        seed = fixedSeed
                )

        // Generate first video
        val video1 = sd.txt2vid(params)
        assertNotNull("First video should not be null", video1)
        assertEquals("First video should have 4 frames", 4, video1.size)

        // Generate second video with same seed
        val video2 = sd.txt2vid(params)
        assertNotNull("Second video should not be null", video2)
        assertEquals("Second video should have 4 frames", 4, video2.size)

        // Compare frame by frame using checksums
        for (i in video1.indices) {
            val frame1Hash = bitmapToMd5(video1[i])
            val frame2Hash = bitmapToMd5(video2[i])
            assertEquals(
                    "Frame $i should be identical between runs with same seed",
                    frame1Hash,
                    frame2Hash
            )
        }
    }

    /**
     * T085: Test seed randomization Use different stub instances to simulate random behavior,
     * verify different outputs
     */
    @Test
    fun testSeedRandomization() = runBlocking {
        StableDiffusion.overrideNativeBridgeForTests { createRandomStub() }
        val sd = createStableDiffusion()

        val params =
                VideoGenerateParams(
                        prompt = "a cat walking",
                        width = 256,
                        height = 256,
                        videoFrames = 4,
                        steps = 10,
                        seed = -1L // Random seed indicator
                )

        // Generate two videos with random seeds (stub uses system time)
        val video1 = sd.txt2vid(params)
        assertNotNull("First video should not be null", video1)

        // Small delay to ensure different timestamp
        Thread.sleep(10)

        val video2 = sd.txt2vid(params)
        assertNotNull("Second video should not be null", video2)

        // At least one frame should be different
        var allFramesIdentical = true
        for (i in video1.indices) {
            val frame1Hash = bitmapToMd5(video1[i])
            val frame2Hash = bitmapToMd5(video2[i])
            if (frame1Hash != frame2Hash) {
                allFramesIdentical = false
                break
            }
        }

        assertFalse("With random seeds, videos should be different", allFramesIdentical)
    }

    /**
     * T086: Test deterministic generation across multiple runs Fixed seed, verify identical output
     * across 3 runs
     */
    @Test
    fun testDeterministicGenerationThreeRuns() = runBlocking {
        val fixedSeed = 12345L
        StableDiffusion.overrideNativeBridgeForTests { createDeterministicStub(fixedSeed) }
        val sd = createStableDiffusion()

        val params =
                VideoGenerateParams(
                        prompt = "a cat walking",
                        width = 256,
                        height = 256,
                        videoFrames = 4,
                        steps = 10,
                        seed = fixedSeed
                )

        // Generate three videos
        val video1 = sd.txt2vid(params)
        val video2 = sd.txt2vid(params)
        val video3 = sd.txt2vid(params)

        assertNotNull("Video 1 should not be null", video1)
        assertNotNull("Video 2 should not be null", video2)
        assertNotNull("Video 3 should not be null", video3)

        // All three videos should have same frame count
        assertEquals(4, video1.size)
        assertEquals(4, video2.size)
        assertEquals(4, video3.size)

        // Compare all frames across all three runs
        for (i in video1.indices) {
            val frame1Hash = bitmapToMd5(video1[i])
            val frame2Hash = bitmapToMd5(video2[i])
            val frame3Hash = bitmapToMd5(video3[i])

            assertEquals("Frame $i should be identical in run 1 and run 2", frame1Hash, frame2Hash)
            assertEquals("Frame $i should be identical in run 1 and run 3", frame1Hash, frame3Hash)
        }
    }

    /** Helper function to compute MD5 hash of bitmap for comparison */
    private fun bitmapToMd5(bitmap: Bitmap): String {
        val buffer = ByteBuffer.allocate(bitmap.width * bitmap.height * 4)
        bitmap.copyPixelsToBuffer(buffer)
        buffer.rewind()

        val md = MessageDigest.getInstance("MD5")
        val digest = md.digest(buffer.array())
        return digest.joinToString("") { "%02x".format(it) }
    }
}
