package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.ImageRuntimeConfig
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.runtime.RuntimeCapabilities
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockkObject
import java.lang.Thread.sleep
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertFalse
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
        ImageClient.resetVideoVulkanBlacklistForTests()
        mockkObject(StableDiffusion.Companion)
        mockkObject(RuntimeCapabilities)
    }

    @After
    fun teardown() {
        ImageClient.resetVideoVulkanBlacklistForTests()
        StableDiffusion.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
        try {
            io.mockk.unmockkObject(StableDiffusion.Companion)
        } catch (_: Throwable) {
        }
        try {
            io.mockk.unmockkObject(RuntimeCapabilities)
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
            StableDiffusion.loadWithRuntimeBackend(
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
            ImageClient.forTesting(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(),
                resolver = DefaultModelRepository(),
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
    fun `image generation keeps gpu backend eligible when preferPerformanceMode is false`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile =
            java.io.File.createTempFile("image-model", ".safetensors", baseDir).apply {
                writeBytes(byteArrayOf(0x01))
            }

        every { RuntimeCapabilities.isStableDiffusionVulkanAvailable() } returns true
        every { RuntimeCapabilities.isStableDiffusionOpenClAvailable() } returns false

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
                ): ByteArray = ByteArray(width * height * 3) { 0x33 }

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
                    Array(videoFrames) { ByteArray(width * height * 3) { 0x33 } }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        val loadBackends = mutableListOf<ComputeBackend>()
        val offloadFlags = mutableListOf<Boolean>()
        val sequentialFlags = mutableListOf<Boolean?>()
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            loadBackends += callArgs[20] as ComputeBackend
            offloadFlags += callArgs[8] as Boolean
            sequentialFlags += callArgs[13] as Boolean?
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            ImageClient.forTesting(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(image = ImageRuntimeConfig(preferPerformanceMode = false)),
                resolver = DefaultModelRepository(),
            )

        try {
            val bitmap =
                client.generate(
                    ImageGenerationRequest(
                        prompt = "test image",
                        width = 256,
                        height = 256,
                        model = ModelSpec.localFile(modelFile),
                    ),
                )

            assertEquals(listOf(ComputeBackend.VULKAN), loadBackends)
            assertEquals(listOf(false), offloadFlags)
            assertEquals(listOf(false), sequentialFlags)
            assertEquals(256, bitmap.width)
            assertEquals(256, bitmap.height)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `video generation retries device lost with Vulkan hard-disabled`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile = java.io.File.createTempFile("wan-model", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile = java.io.File.createTempFile("wan-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5File = java.io.File.createTempFile("umt5", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        every { RuntimeCapabilities.isStableDiffusionVulkanAvailable() } returns true

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
                    if (handle == 1L) {
                        throw RuntimeException("vk::Queue::submit: ErrorDeviceLost")
                    }
                    return Array(videoFrames) { ByteArray(width * height * 3) { 0x11 } }
                }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        val loadBackends = mutableListOf<ComputeBackend>()
        var loadCount = 0L
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            loadBackends += callArgs[20] as ComputeBackend
            loadCount += 1
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(loadCount).apply {
                updateModelMetadata(
                    VideoModelMetadata(
                        architecture = "Wan 2.1 T2V",
                        modelType = null,
                        parameterCount = "1.3B",
                        mobileSupported = true,
                        tags = setOf("wan-model"),
                        filename = modelFile.name,
                    ),
                )
            }
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            ImageClient.forTesting(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(image = ImageRuntimeConfig(preferPerformanceMode = true)),
                resolver = DefaultModelRepository(),
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
                    model = ModelSpec.localFile(modelFile),
                    vae = ModelSpec.localFile(vaeFile),
                    textEncoder = ModelSpec.localFile(t5File),
                ),
            ).collect { event ->
                if (event is GenerationStreamEvent.Completed) {
                    frames = event.frames
                }
            }

            val completedFrames = requireNotNull(frames)
            assertEquals(listOf(ComputeBackend.VULKAN, ComputeBackend.CPU), loadBackends)
            assertEquals(5, completedFrames.size)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `video device loss blacklists Vulkan for later generations in same process`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile = java.io.File.createTempFile("wan-model", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile = java.io.File.createTempFile("wan-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5File = java.io.File.createTempFile("umt5", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        every { RuntimeCapabilities.isStableDiffusionVulkanAvailable() } returns true

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
                    if (handle == 1L) {
                        throw RuntimeException("vk::Queue::submit: ErrorDeviceLost")
                    }
                    return Array(videoFrames) { ByteArray(width * height * 3) { 0x22 } }
                }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        val loadBackends = mutableListOf<ComputeBackend>()
        var loadCount = 0L
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            loadBackends += callArgs[20] as ComputeBackend
            loadCount += 1
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(loadCount).apply {
                updateModelMetadata(
                    VideoModelMetadata(
                        architecture = "Wan 2.1 T2V",
                        modelType = null,
                        parameterCount = "1.3B",
                        mobileSupported = true,
                        tags = setOf("wan-model"),
                        filename = modelFile.name,
                    ),
                )
            }
        }

        suspend fun runGeneration(client: ImageClient): List<Bitmap> {
            var frames: List<Bitmap>? = null
            client.generateVideo(
                VideoGenerationRequest(
                    prompt = "test prompt",
                    width = 256,
                    height = 256,
                    videoFrames = 5,
                    steps = 20,
                    model = ModelSpec.localFile(modelFile),
                    vae = ModelSpec.localFile(vaeFile),
                    textEncoder = ModelSpec.localFile(t5File),
                ),
            ).collect { event ->
                if (event is GenerationStreamEvent.Completed) {
                    frames = event.frames
                }
            }
            return requireNotNull(frames)
        }

        val edgeScope1 = LLMEdgeScope(this, 1)
        val edgeScope2 = LLMEdgeScope(this, 1)
        val client1 =
            ImageClient.forTesting(
                context = context,
                scope = edgeScope1,
                config = LLMEdgeConfig(image = ImageRuntimeConfig(preferPerformanceMode = true)),
                resolver = DefaultModelRepository(),
            )
        val client2 =
            ImageClient.forTesting(
                context = context,
                scope = edgeScope2,
                config = LLMEdgeConfig(image = ImageRuntimeConfig(preferPerformanceMode = true)),
                resolver = DefaultModelRepository(),
            )

        try {
            assertEquals(5, runGeneration(client1).size)
            assertEquals(5, runGeneration(client2).size)
            assertEquals(
                listOf(ComputeBackend.VULKAN, ComputeBackend.CPU, ComputeBackend.CPU),
                loadBackends,
            )
        } finally {
            client1.close()
            client2.close()
            edgeScope1.close()
            edgeScope2.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `image generation keeps easycache disabled unless explicitly requested`() = runTest {
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
            StableDiffusion.loadWithRuntimeBackend(
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
            ImageClient.forTesting(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(),
                resolver = DefaultModelRepository(),
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
            assertEquals(listOf(false), easyCacheFlags)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `image generation reuses cached runtime across requests and reports warm metrics`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile = java.io.File.createTempFile("cached-image-model", ".gguf", context.filesDir).apply { writeBytes(byteArrayOf(0x01)) }

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
                ): ByteArray = ByteArray(width * height * 3) { 0x33 }

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
            }
        }

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            sleep(5)
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            ImageClient.forTesting(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(),
                resolver = DefaultModelRepository(),
            )

        try {
            val metrics = mutableListOf<io.aatricks.llmedge.image.diffusion.ImageRequestMetrics>()
            repeat(2) {
                val bitmap =
                    client.generate(
                        ImageGenerationRequest(
                            prompt = "cached prompt",
                            width = 64,
                            height = 64,
                            model = ModelSpec.localFile(modelFile),
                        ),
                    )
                assertEquals(64, bitmap.width)
                assertEquals(64, bitmap.height)
                metrics += requireNotNull(client.getLastGenerationMetrics()?.imageRequestMetrics)
            }

            coVerify(exactly = 1) {
                StableDiffusion.loadWithRuntimeBackend(
                    any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
                )
            }
            assertFalse(metrics.first().cacheHit)
            assertTrue(metrics.first().modelLoadMs > 0L)
            assertTrue(metrics.first().runtimeAcquireMs >= metrics.first().modelLoadMs)
            assertTrue(metrics.last().cacheHit)
            assertEquals(0L, metrics.last().modelLoadMs)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `sequential video generation keeps easycache disabled unless explicitly requested`() = runTest {
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
            StableDiffusion.loadWithRuntimeBackend(
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
            ImageClient.forTesting(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(),
                resolver = DefaultModelRepository(),
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

            assertEquals(listOf(false), easyCacheFlags)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `image generation retries same backend with flash attention disabled before backend fallback`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile =
            java.io.File.createTempFile("flash-fallback-image", ".safetensors", baseDir).apply {
                writeBytes(byteArrayOf(0x01))
            }

        every { RuntimeCapabilities.isStableDiffusionVulkanAvailable() } returns true
        every { RuntimeCapabilities.isStableDiffusionOpenClAvailable() } returns false

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
                ): ByteArray = ByteArray(width * height * 3) { 0x55 }

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
            }
        }

        val flashFlags = mutableListOf<Boolean>()
        val loadBackends = mutableListOf<ComputeBackend>()
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            val flashAttn = callArgs[11] as Boolean
            val backend = callArgs[20] as ComputeBackend
            flashFlags += flashAttn
            loadBackends += backend
            if (flashAttn) {
                throw IllegalStateException("flash init failed")
            }
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            ImageClient.forTesting(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(image = ImageRuntimeConfig(preferPerformanceMode = true)),
                resolver = DefaultModelRepository(),
            )

        try {
            val bitmap =
                client.generate(
                    ImageGenerationRequest(
                        prompt = "test image",
                        width = 256,
                        height = 256,
                        flashAttention = true,
                        model = ModelSpec.localFile(modelFile),
                    ),
                )

            assertEquals(256, bitmap.width)
            assertEquals(256, bitmap.height)
            assertEquals(listOf(ComputeBackend.VULKAN, ComputeBackend.VULKAN), loadBackends)
            assertEquals(listOf(true, false), flashFlags)
            val metrics = requireNotNull(client.getLastGenerationMetrics())
            val requestMetrics = requireNotNull(metrics.imageRequestMetrics)
            assertFalse(requestMetrics.cacheHit)
            assertEquals(false, requestMetrics.flashAttentionEnabled)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `image generation preserves lora prompt tags and passes image lora load options`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile =
            java.io.File.createTempFile("image-model-lora", ".safetensors", baseDir).apply {
                writeBytes(byteArrayOf(0x01))
            }
        val loraDir = java.io.File(baseDir, "test-image-lora-dir").apply { mkdirs() }
        val observedPrompts = mutableListOf<String>()
        val observedLoraDirs = mutableListOf<String?>()

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
                    observedPrompts += prompt
                    return ByteArray(width * height * 3) { 0x44 }
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
            }
        }

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            observedLoraDirs += callArgs[18] as String?
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            ImageClient.forTesting(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(image = ImageRuntimeConfig(preferPerformanceMode = true)),
                resolver = DefaultModelRepository(),
            )

        try {
            val bitmap =
                client.generate(
                    ImageGenerationRequest(
                        prompt = "portrait <lora:detail-tweaker:1.0> test",
                        negative = "bad anatomy <lora:detail-tweaker:0.5>",
                        width = 128,
                        height = 128,
                        loraModelDir = loraDir.absolutePath,
                        model = ModelSpec.localFile(modelFile),
                    ),
                )

            assertEquals(128, bitmap.width)
            assertEquals(128, bitmap.height)
            assertEquals(listOf(loraDir.absolutePath), observedLoraDirs)
            assertEquals(listOf("portrait <lora:detail-tweaker:1.0> test"), observedPrompts)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }
}
