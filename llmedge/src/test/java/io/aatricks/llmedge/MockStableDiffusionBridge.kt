package io.aatricks.llmedge

import android.graphics.Bitmap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback

/**
 * Configurable mock implementation of StableDiffusion.NativeBridge for testing.
 * Allows different test scenarios to be simulated.
 */
open class MockStableDiffusionBridge : StableDiffusion.NativeBridge {
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
    ): ByteArray? {
        if (isCancelled.get()) {
            isCancelled.set(false)
            return null
        }
        val callback = activeProgressCallback.get()
        if (callback != null && steps > 0) {
            for (step in 1..steps) {
                if (progressCallbackDelayMs > 0) {
                    Thread.sleep(progressCallbackDelayMs)
                }
                if (isCancelled.get()) {
                    isCancelled.set(false)
                    return null
                }
                callback.onProgress(step, steps, 0, 1, 1.0f)
            }
        }
        return ByteArray(width * height * 3) { 0 }
    }

    // Configuration options
    var shouldFailTxt2Vid = false
    var txt2VidResultFrames = 8
    var txt2VidFrameWidth = 256
    var txt2VidFrameHeight = 256
    var progressCallbackDelayMs = 10L

    // Tracking for verification
    val txt2VidCalls = mutableListOf<Txt2VidCall>()
    val setProgressCallbackCalls = mutableListOf<Pair<Long, VideoProgressCallback?>>()
    val cancelGenerationCalls = mutableListOf<Long>()
    val upscaleCalls = mutableListOf<UpscaleCall>()

    // State for progress simulation
    private val activeProgressCallback = AtomicReference<VideoProgressCallback?>()
    private val isCancelled = AtomicBoolean(false)

    data class Txt2VidCall(
        val handle: Long,
        val prompt: String,
        val negative: String,
        val width: Int,
        val height: Int,
        val videoFrames: Int,
        val steps: Int,
        val cfg: Float,
        val seed: Long,
        val sampleMethod: SampleMethod,
        val scheduler: Scheduler,
        val strength: Float,
        val initImage: ByteArray?,
        val initWidth: Int,
        val initHeight: Int,
        val vaceStrength: Float,
        val easyCacheEnabled: Boolean,
        val easyCacheReuseThreshold: Float,
        val easyCacheStartPercent: Float,
        val easyCacheEndPercent: Float
    )

    data class UpscaleCall(
        val esrganPath: String,
        val nThreads: Int,
        val tileSize: Int,
        val backend: String,
        val width: Int,
        val height: Int,
        val factor: Int
    )

    override fun upscale(
        esrganPath: String,
        nThreads: Int,
        tileSize: Int,
        backend: String,
        pixels: IntArray,
        width: Int,
        height: Int,
        factor: Int,
        outDims: IntArray,
    ): IntArray? {
        upscaleCalls.add(
            UpscaleCall(
                esrganPath = esrganPath,
                nThreads = nThreads,
                tileSize = tileSize,
                backend = backend,
                width = width,
                height = height,
                factor = factor
            )
        )
        val factorUsed = if (factor == 0) 4 else factor
        val outW = width * factorUsed
        val outH = height * factorUsed
        if (outDims.size >= 2) {
            outDims[0] = outW
            outDims[1] = outH
        }
        return IntArray(outW * outH) { 0xFF000000.toInt() }
    }

    override fun upscale(
        esrganPath: String,
        nThreads: Int,
        tileSize: Int,
        backend: String,
        pixels: IntArray,
        width: Int,
        height: Int,
        factor: Int,
        outDims: IntArray,
        progress: VideoProgressCallback?,
    ): IntArray? {
        for (i in 1..4) {
            progress?.onProgress(i, 4, 0, 0, 0.5f)
        }
        return upscale(
            esrganPath = esrganPath,
            nThreads = nThreads,
            tileSize = tileSize,
            backend = backend,
            pixels = pixels,
            width = width,
            height = height,
            factor = factor,
            outDims = outDims
        )
    }

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
        // Track the call
        txt2VidCalls.add(Txt2VidCall(
            handle, prompt, negative, width, height, videoFrames, steps, cfg, seed,
            sampleMethod, scheduler, strength, initImage, initWidth, initHeight, vaceStrength,
            easyCacheEnabled, easyCacheReuseThreshold, easyCacheStartPercent, easyCacheEndPercent
        ))

        // Simulate failure if configured
        if (shouldFailTxt2Vid) {
            return null
        }

        // Simulate cancellation
        if (isCancelled.get()) {
            isCancelled.set(false) // Reset for next call
            return null
        }

        // Simulate progress callbacks
        val callback = activeProgressCallback.get()
        if (callback != null && steps > 0) {
            // Simulate progress updates
            for (step in 1..steps) {
                Thread.sleep(progressCallbackDelayMs)
                // Respect cancellation during progress updates
                if (isCancelled.get()) {
                    isCancelled.set(false)
                    return null
                }
                callback.onProgress(step, steps, 0, videoFrames, 1.0f)
            }
        }

        // Generate mock frame data
        return Array(videoFrames) { frameIndex ->
            // Create RGB data for each frame (width * height * 3 bytes)
            ByteArray(width * height * 3) { byteIndex ->
                // Create some variation based on frame index and byte position
                ((frameIndex * 17 + byteIndex * 13) % 256).toByte()
            }
        }
    }

    override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) {
        setProgressCallbackCalls.add(handle to callback)
        activeProgressCallback.set(callback)
    }

    override fun cancelGeneration(handle: Long) {
        cancelGenerationCalls.add(handle)
        isCancelled.set(true)
    }

    // Utility methods for test configuration
    fun reset() {
        shouldFailTxt2Vid = false
        txt2VidResultFrames = 8
        txt2VidFrameWidth = 256
        txt2VidFrameHeight = 256
        progressCallbackDelayMs = 10L

        txt2VidCalls.clear()
        setProgressCallbackCalls.clear()
        cancelGenerationCalls.clear()
        upscaleCalls.clear()

        activeProgressCallback.set(null)
        isCancelled.set(false)
    }

    fun configureForFailure() {
        shouldFailTxt2Vid = true
    }

    fun configureFrameDimensions(width: Int, height: Int) {
        txt2VidFrameWidth = width
        txt2VidFrameHeight = height
    }

    fun configureFrameCount(count: Int) {
        txt2VidResultFrames = count
    }

    fun disableProgressDelays() {
        progressCallbackDelayMs = 0L
    }

    override fun precomputeCondition(
        handle: Long,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        clipSkip: Int,
    ): PrecomputedCondition? = null
}
