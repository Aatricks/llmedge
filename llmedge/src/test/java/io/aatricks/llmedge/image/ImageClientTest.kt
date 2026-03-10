package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
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
        val modelFile = java.io.File.createTempFile("wan-model", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile = java.io.File.createTempFile("wan-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5File = java.io.File.createTempFile("umt5", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

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
                ): Array<ByteArray> =
                    Array(videoFrames) { ByteArray(width * height * 3) { ((it + 1) % 255).toByte() } }

                override fun precomputeCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    clipSkip: Int,
                ): PrecomputedCondition =
                    PrecomputedCondition(
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
                    sampleMethod: SampleMethod,
                    scheduler: Scheduler,
                    strength: Float,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                    cond: PrecomputedCondition?,
                    uncond: PrecomputedCondition?,
                    vaceStrength: Float,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
                ): Array<ByteArray> =
                    Array(videoFrames) { ByteArray(width * height * 3) { 5 } }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

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
                VideoModelMetadata(
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

    @Test
    fun `image generation auto-enables easycache for supported models`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile = java.io.File.createTempFile("flux-model", ".gguf", context.filesDir).apply { writeBytes(byteArrayOf(0x01)) }
        val easyCacheFlags = mutableListOf<Boolean>()

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
                ): ByteArray {
                    easyCacheFlags += easyCacheEnabled
                    return ByteArray(width * height * 3) { 0x7F }
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
                ): Array<ByteArray>? = null

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

        coEvery {
            StableDiffusion.load(
                any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L).apply {
                updateModelMetadata(
                    VideoModelMetadata(
                        architecture = "Flux",
                        modelType = null,
                        parameterCount = null,
                        mobileSupported = true,
                        tags = setOf("flux"),
                        filename = modelFile.name,
                    ),
                )
            }
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
            val bitmap =
                client.generate(
                    ImageGenerationRequest(
                        prompt = "test prompt",
                        width = 64,
                        height = 64,
                        model = ModelSpec.localFile(modelFile),
                    ),
                )

            assertEquals(64, bitmap.width)
            assertEquals(64, bitmap.height)
            assertEquals(listOf(true), easyCacheFlags)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `sequential video generation auto-enables easycache for supported models`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile = java.io.File.createTempFile("wan-model", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile = java.io.File.createTempFile("wan-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5File = java.io.File.createTempFile("umt5", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val easyCacheFlags = mutableListOf<Boolean>()

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
                ): ByteArray = ByteArray(width * height * 3) { 0 }

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

                override fun precomputeCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    clipSkip: Int,
                ): PrecomputedCondition =
                    PrecomputedCondition(
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
                    sampleMethod: SampleMethod,
                    scheduler: Scheduler,
                    strength: Float,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                    cond: PrecomputedCondition?,
                    uncond: PrecomputedCondition?,
                    vaceStrength: Float,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
                ): Array<ByteArray> {
                    easyCacheFlags += easyCacheEnabled
                    return Array(videoFrames) { ByteArray(width * height * 3) { 5 } }
                }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit
                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        coEvery {
            StableDiffusion.load(
                any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L).apply {
                val modelPathArg = callArgs[3] as String?
                val metadata =
                    if (modelPathArg == modelFile.absolutePath) {
                        VideoModelMetadata(
                            architecture = "Wan 2.1 T2V",
                            modelType = null,
                            parameterCount = "1.3B",
                            mobileSupported = true,
                            tags = setOf("wan-model"),
                            filename = modelFile.name,
                        )
                    } else {
                        VideoModelMetadata(
                            architecture = "text-encoder",
                            modelType = null,
                            parameterCount = null,
                            mobileSupported = true,
                            tags = emptySet(),
                            filename = t5File.name,
                        )
                    }
                updateModelMetadata(metadata)
            }
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
            client.generateVideo(
                VideoGenerationRequest(
                    prompt = "test prompt",
                    width = 256,
                    height = 256,
                    videoFrames = 5,
                    steps = 20,
                    forceSequentialLoad = true,
                    model = ModelSpec.localFile(modelFile),
                    vae = ModelSpec.localFile(vaeFile),
                    textEncoder = ModelSpec.localFile(t5File),
                ),
            ).collect()

            assertEquals(listOf(true), easyCacheFlags)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }
}
