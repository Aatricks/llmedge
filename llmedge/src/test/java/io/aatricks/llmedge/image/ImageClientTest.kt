package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.StableDiffusion
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.DefaultModelResolver
import io.aatricks.llmedge.model.ModelSpec
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.mockkObject
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ImageClientTest {
    @Before
    fun setup() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        mockkObject(StableDiffusion.Companion)
    }

    @After
    fun teardown() {
        StableDiffusion.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
        try {
            io.mockk.unmockkObject(StableDiffusion.Companion)
        } catch (_: Throwable) {
        }
        clearAllMocks()
    }

    @Test
    fun `sequential video generation loads text encoder before diffusion model`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile = java.io.File.createTempFile("wan-model", ".gguf", baseDir)
        val vaeFile = java.io.File.createTempFile("wan-vae", ".safetensors", baseDir)
        val t5File = java.io.File.createTempFile("umt5", ".gguf", baseDir)

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
                ): ByteArray = ByteArray(3 * width * height) { 0 }

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
                    sampleMethod: StableDiffusion.SampleMethod,
                    scheduler: StableDiffusion.Scheduler,
                    strength: Float,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                    vaceStrength: Float,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
                ): Array<ByteArray> =
                    Array(videoFrames) { ByteArray(width * height * 3) { ((it + 1) % 255).toByte() } }

                override fun precomputeCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    clipSkip: Int,
                ): StableDiffusion.PrecomputedCondition =
                    StableDiffusion.PrecomputedCondition(
                        cCrossAttn = floatArrayOf(1.0f),
                        cCrossAttnDims = intArrayOf(1, 1),
                        cVector = floatArrayOf(1.0f),
                        cVectorDims = intArrayOf(1, 1),
                        cConcat = floatArrayOf(1.0f),
                        cConcatDims = intArrayOf(1, 1),
                    )

                override fun txt2vidWithPrecomputedCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    videoFrames: Int,
                    steps: Int,
                    cfg: Float,
                    seed: Long,
                    sampleMethod: StableDiffusion.SampleMethod,
                    scheduler: StableDiffusion.Scheduler,
                    strength: Float,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                    cond: StableDiffusion.PrecomputedCondition?,
                    uncond: StableDiffusion.PrecomputedCondition?,
                    vaceStrength: Float,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
                ): Array<ByteArray> =
                    Array(videoFrames) { ByteArray(width * height * 3) { 5 } }

                override fun setProgressCallback(handle: Long, callback: StableDiffusion.VideoProgressCallback?) = Unit

                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        val observedLoads = mutableListOf<Triple<String?, String?, String?>>()
        coEvery {
            StableDiffusion.load(
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            observedLoads.add(
                Triple(
                    callArgs[3] as String?,
                    callArgs[4] as String?,
                    callArgs[5] as String?,
                ),
            )
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            val instance = constructor.newInstance(1L)
            instance.updateModelMetadata(
                StableDiffusion.VideoModelMetadata(
                    architecture = "Wan 2.1 T2V",
                    modelType = null,
                    parameterCount = "1.3B",
                    mobileSupported = true,
                    tags = setOf("wan-model"),
                    filename = modelFile.name,
                ),
            )
            instance
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            ImageClient(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(),
                resolver = DefaultModelResolver(),
            )

        try {
            var frames: List<Bitmap>? = null
            client.generateVideo(
                    VideoGenerationRequest(
                        prompt = "test prompt",
                        width = 256,
                        height = 256,
                        videoFrames = 5,
                        steps = 20,
                        cfgScale = 7.0f,
                        seed = 123L,
                        forceSequentialLoad = true,
                        model = ModelSpec.localFile(modelFile),
                        vae = ModelSpec.localFile(vaeFile),
                        textEncoder = ModelSpec.localFile(t5File),
                    ),
                )
                .collect { event ->
                    if (event is GenerationStreamEvent.Completed) {
                        frames = event.frames
                    }
                }
            val completedFrames = requireNotNull(frames)

            assertNotNull(completedFrames)
            assertTrue(observedLoads.size >= 2)
            assertTrue(observedLoads.any { it.first == t5File.absolutePath })
            assertTrue(observedLoads.any { it.first == modelFile.absolutePath })
            assertTrue(observedLoads.any { it.first == modelFile.absolutePath && it.third == null })
            assertEquals(5, completedFrames.size)
            completedFrames.forEach {
                assertEquals(256, it.width)
                assertEquals(256, it.height)
                assertEquals(Bitmap.Config.ARGB_8888, it.config)
            }
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }
}
