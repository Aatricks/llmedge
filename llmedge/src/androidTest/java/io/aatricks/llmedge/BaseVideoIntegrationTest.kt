package io.aatricks.llmedge

import android.graphics.Color
import org.junit.After
import org.junit.Before

abstract class BaseVideoIntegrationTest {
    companion object {
        @Suppress("unused")
        private val disableNativeLoadForAndroidTests = run {
            System.setProperty("llmedge.disableNativeLoad", "true")
            true
        }
    }

    @Before
    fun baseSetUp() {
        StableDiffusion.enableNativeBridgeForTests()
        // Provide a default stubbed native bridge for all video tests.
        // Individual tests can override this with their own behavior.
        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
                private var cb: VideoProgressCallback? = null
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
                    // Simple black image stub
                    return ByteArray((width * height * 3).coerceAtLeast(0)) { 0 }
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
                    // Deterministic grayscale frames
                    val frames =
                            Array(videoFrames.coerceAtLeast(0)) { index ->
                                val bytes = ByteArray((width * height * 3).coerceAtLeast(0))
                                var i = 0
                                val v = (index * 0x10) and 0xFF
                                val b = v.toByte()
                                while (i < bytes.size) {
                                    bytes[i++] = b
                                    bytes[i++] = b
                                    bytes[i++] = b
                                }
                                bytes
                            }
                    // Emit basic progress per frame
                    for (i in 0 until videoFrames) {
                        cb?.onProgress(i + 1, steps, i + 1, videoFrames, 0.01f)
                    }
                    return frames
                }
                override fun setProgressCallback(
                        handle: Long,
                        callback: VideoProgressCallback?
                ) {
                    cb = callback
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
    }

    @After
    fun baseTearDown() {
        StableDiffusion.resetNativeBridgeForTests()
    }

    protected fun createStableDiffusion(): StableDiffusion {
        val constructor =
                StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        return constructor.newInstance(1L).apply {
            updateModelMetadata(
                    VideoModelMetadata(
                            architecture = "wan",
                            modelType = "t2v",
                            parameterCount = "1.3B",
                            mobileSupported = true,
                            filename = "wan-test.gguf",
                            tags = setOf("wan", "video"),
                    ),
            )
        }
    }

    protected fun buildFrames(frameCount: Int, width: Int, height: Int): Array<ByteArray> {
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
