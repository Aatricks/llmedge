package io.aatricks.llmedge

import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.assertSame
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@Suppress("unused")
private val disableNativeLoadForTxt2VidTests = run {
    System.setProperty("llmedge.disableNativeLoad", "true")
    true
}

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StableDiffusionTxt2VidTest {
    companion object {
        private const val TEST_DIMENSION = 256
        private const val TEST_FRAMES = 5  // Minimum 5 frames required for Wan model
    }

    @Before
    fun setUp() {
        StableDiffusion.enableNativeBridgeForTests()
    }

    @After
    fun tearDown() {
        StableDiffusion.resetNativeBridgeForTests()
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidConvertsFramesToBitmaps() = runTest {
        val frames = buildFrames(frameCount = TEST_FRAMES, width = TEST_DIMENSION, height = TEST_DIMENSION)
        val progressRegistrations = mutableListOf<VideoProgressCallback?>()
        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
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
                ): Array<ByteArray> = frames.map { it.clone() }.toTypedArray()

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) {
                    progressRegistrations += callback
                }

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
        val sd = testableStableDiffusion()

        val callback = VideoProgressCallback { _, _, _, _, _ -> }
        val params = VideoGenerateParams(
            prompt = "go",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = TEST_FRAMES,
        )

        val bitmaps = sd.txt2vid(params, callback)

        assertEquals(TEST_FRAMES, bitmaps.size)
        assertEquals(TEST_DIMENSION, bitmaps.first().width)
        assertEquals(Color.rgb(0x10, 0x10, 0x10), bitmaps[1].getPixel(0, 0))
        assertNotNull(sd.getLastGenerationMetrics())
        assertTrue(progressRegistrations.contains(callback))
        assertTrue(progressRegistrations.any { it == null })
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidRejectsFrameCountBelowMinimum() = runTest {
        val sd = testableStableDiffusion()

        val params = VideoGenerateParams(
            prompt = "wan",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = 2,
        )

        val failure = runCatching { sd.txt2vid(params) }

        assertTrue(failure.exceptionOrNull() is IllegalArgumentException)
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidRejectsZeroFrameRequests() = runTest {
        val sd = testableStableDiffusion()

        val params = VideoGenerateParams(
            prompt = "wan",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = 0,
        )

        val failure = runCatching { sd.txt2vid(params) }

        assertTrue(failure.exceptionOrNull() is IllegalArgumentException)
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidRejectsSeedBelowNegativeOne() = runTest {
        val sd = testableStableDiffusion()

        val params = VideoGenerateParams(
            prompt = "wan",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = TEST_FRAMES,
            seed = -5,
        )

        val failure = runCatching { sd.txt2vid(params) }

        assertTrue(failure.exceptionOrNull() is IllegalArgumentException)
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidPassesInitImageBytesToNativeLayer() = runTest {
        val frames = buildFrames(frameCount = TEST_FRAMES, width = TEST_DIMENSION, height = TEST_DIMENSION)
        var capturedInitWidth = -1
        var capturedInitHeight = -1
        var capturedBytes: ByteArray? = null
        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
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
                ): Array<ByteArray> {
                    capturedBytes = initImage
                    capturedInitWidth = initWidth
                    capturedInitHeight = initHeight
                    return frames.map { it.clone() }.toTypedArray()
                }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

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
        val sd = testableStableDiffusion()

        val initBitmap = Bitmap.createBitmap(TEST_DIMENSION, TEST_DIMENSION, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.CYAN)
        }
        val params = VideoGenerateParams(
            prompt = "wan",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = TEST_FRAMES,
            initImage = initBitmap,
        )

        sd.txt2vid(params)

        val bytes = requireNotNull(capturedBytes)
        assertEquals(TEST_DIMENSION * TEST_DIMENSION * 3, bytes.size)
        assertEquals(TEST_DIMENSION, capturedInitWidth)
        assertEquals(TEST_DIMENSION, capturedInitHeight)
        assertEquals(Color.red(Color.CYAN), bytes[0].toInt() and 0xFF)
        assertEquals(Color.green(Color.CYAN), bytes[1].toInt() and 0xFF)
        assertEquals(Color.blue(Color.CYAN), bytes[2].toInt() and 0xFF)
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidThrowsWhenNativeReturnsNoFrames() = runTest {
        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
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
                ): Array<ByteArray> = emptyArray()

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

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
        val sd = testableStableDiffusion()

        val params = VideoGenerateParams(
            prompt = "wan",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = TEST_FRAMES,
        )

        val result = runCatching { sd.txt2vid(params) }
        assertTrue(result.exceptionOrNull() is IllegalStateException)
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidPropagatesCancellationWhenFlagged() = runTest {
        StableDiffusion.overrideNativeBridgeForTests { instance ->
            object : StableDiffusion.NativeBridge {
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
                    instance.cancelGeneration()
                    throw RuntimeException("native aborted")
                }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

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
        val sd = testableStableDiffusion()

        val params = VideoGenerateParams(
            prompt = "wan",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = TEST_FRAMES,
        )

        val result = runCatching { sd.txt2vid(params) }
        assertTrue(result.exceptionOrNull() is CancellationException)
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidSkipsProgressRegistrationWhenCallbackNull() = runTest {
        val frames = buildFrames(frameCount = TEST_FRAMES, width = TEST_DIMENSION, height = TEST_DIMENSION)
        val setProgressInvocations = mutableListOf<VideoProgressCallback?>()
        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
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
                ): Array<ByteArray>? = frames.map { it.clone() }.toTypedArray()

                override fun setProgressCallback(
                    handle: Long,
                    callback: VideoProgressCallback?,
                ) {
                    setProgressInvocations += callback
                }

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
        val sd = testableStableDiffusion()
        val params = VideoGenerateParams(
            prompt = "wan",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = TEST_FRAMES,
        )

        sd.txt2vid(params)

        assertTrue(setProgressInvocations.isEmpty())
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun txt2vidRestoresCachedProgressCallbackAfterFailure() = runTest {
        val setProgressInvocations = mutableListOf<VideoProgressCallback?>()
        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
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
                ): Array<ByteArray>? = null

                override fun setProgressCallback(
                    handle: Long,
                    callback: VideoProgressCallback?,
                ) {
                    setProgressInvocations += callback
                }

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
        val sd = testableStableDiffusion()
        val cachedCallback = VideoProgressCallback { _, _, _, _, _ -> }
        val tempCallback = VideoProgressCallback { _, _, _, _, _ -> }
        sd.setProgressCallback(cachedCallback)

        val params = VideoGenerateParams(
            prompt = "wan",
            width = TEST_DIMENSION,
            height = TEST_DIMENSION,
            videoFrames = TEST_FRAMES,
        )

        val result = runCatching { sd.txt2vid(params, tempCallback) }

        assertTrue(result.exceptionOrNull() is IllegalStateException)
        assertEquals(3, setProgressInvocations.size)
        assertSame(cachedCallback, setProgressInvocations[0])
        assertSame(tempCallback, setProgressInvocations[1])
        assertSame(cachedCallback, setProgressInvocations[2])
    }

    private fun testableStableDiffusion(): StableDiffusion {
        val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        val instance = constructor.newInstance(1L)
        instance.updateModelMetadata(VideoModelMetadata(
            architecture = "wan",
            modelType = "t2v",
            parameterCount = "1.3B",
            mobileSupported = true,
            tags = setOf("wan", "video"),
            filename = "test-model.gguf"
        ))
        return instance
    }

    private fun buildFrames(frameCount: Int, width: Int, height: Int): Array<ByteArray> {
        return Array(frameCount) { index ->
            val color = Color.rgb(index * 0x10, index * 0x10, index * 0x10)
            val bytes = ByteArray(width * height * 3)
            var i = 0
            val r = Color.red(color).toByte()
            val g = Color.green(color).toByte()
            val b = Color.blue(color).toByte()
            while (i < bytes.size) {
                bytes[i++] = r
                bytes[i++] = g
                bytes[i++] = b
            }
            bytes
        }
    }
}
