package io.aatricks.llmedge.image

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.ImageRuntimeConfig
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.runtime.RuntimeCapabilities
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import io.aatricks.llmedge.image.diffusion.StableDiffusionLoadRequest
import io.aatricks.llmedge.image.diffusion.StableDiffusionNativeLoadRequest
import io.aatricks.llmedge.image.diffusion.StableDiffusionAssetRequest
import io.aatricks.llmedge.image.diffusion.StableDiffusionRuntimeRequest
import io.aatricks.llmedge.image.diffusion.StableDiffusionBackendRequest
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths
import io.aatricks.llmedge.image.diffusion.internal.StableDiffusionLoader
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
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
    fun `video diffusion artifact routes to native diffusion model path`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile =
            java.io.File.createTempFile("wan22", ".gguf", context.filesDir).apply {
                writeBytes(byteArrayOf(0x01))
            }
        val model =
            ModelSpec.localFile(
                modelFile,
                ModelHints(artifactKind = ModelArtifactKind.DIFFUSION_MODEL),
            )
        var observedModelPath: String? = "unset"
        var observedDiffusionModelPath: String? = "unset"
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
            )
        } coAnswers {
            observedModelPath = it.invocation.args[3] as String?
            observedDiffusionModelPath = it.invocation.args[21] as String?
            mockk(relaxed = true)
        }

        val loaded =
            DiffusionRuntimeLoader(context, DefaultModelRepository()).load(
                spec = DiffusionRuntimeSpec(role = DiffusionRuntimeRole.VIDEO, model = model),
                options =
                    DiffusionLoadOptions(
                        subsystem = ComputeSubsystem.VIDEO,
                        allowGpu = false,
                        nThreads = 1,
                        offloadToCpu = true,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = true,
                        preferPerformanceMode = false,
                    ),
                backend = ComputeBackend.CPU,
            )
        try {
            assertEquals(null, observedModelPath)
            assertEquals(modelFile.absolutePath, observedDiffusionModelPath)
        } finally {
            loaded.close()
        }
    }

    @Test
    fun `image diffusion artifact routes to native model path when diffusionModelOnly is false`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile =
            java.io.File.createTempFile("classic-image", ".gguf", context.filesDir).apply {
                writeBytes(byteArrayOf(0x01))
            }
        val model =
            ModelSpec.localFile(
                modelFile,
                ModelHints(artifactKind = ModelArtifactKind.DIFFUSION_MODEL),
            )
        var observedModelPath: String? = "unset"
        var observedDiffusionModelPath: String? = "unset"
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
            )
        } coAnswers {
            observedModelPath = it.invocation.args[3] as String?
            observedDiffusionModelPath = it.invocation.args[21] as String?
            mockk(relaxed = true)
        }

        val loaded =
            DiffusionRuntimeLoader(context, DefaultModelRepository()).load(
                spec = DiffusionRuntimeSpec(role = DiffusionRuntimeRole.IMAGE, model = model, diffusionModelOnly = false),
                options =
                    DiffusionLoadOptions(
                        subsystem = ComputeSubsystem.IMAGE,
                        allowGpu = false,
                        nThreads = 1,
                        offloadToCpu = true,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = true,
                        preferPerformanceMode = false,
                    ),
                backend = ComputeBackend.CPU,
            )
        try {
            assertEquals(modelFile.absolutePath, observedModelPath)
            assertEquals(null, observedDiffusionModelPath)
        } finally {
            loaded.close()
        }
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
                any(),
                any(),
                any(),
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
    fun `split-model image request routes DiT to diffusionModelPath and encoder to llmPath`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val ditFile =
            java.io.File.createTempFile("flux2-dit", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile =
            java.io.File.createTempFile("flux2-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val encFile =
            java.io.File.createTempFile("qwen3-enc", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

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
                ): Array<ByteArray> = Array(videoFrames) { ByteArray(width * height * 3) { 0x33 } }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        var observedModelPath: String? = "unset"
        var observedDiffusionModelPath: String? = "unset"
        var observedLlmPath: String? = "unset"
        var observedVaePath: String? = "unset"
        var observedOffload: Boolean? = null
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            observedModelPath = callArgs[3] as String?
            observedVaePath = callArgs[4] as String?
            observedOffload = callArgs[8] as Boolean
            observedDiffusionModelPath = callArgs[21] as String?
            observedLlmPath = callArgs[22] as String?
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
            client.generate(
                ImageGenerationRequest(
                    prompt = "a red fox in snow",
                    width = 256,
                    height = 256,
                    model = ModelSpec.localFile(ditFile),
                    vae = ModelSpec.localFile(vaeFile),
                    textEncoder = ModelSpec.localFile(encFile),
                    splitDiffusionModel = true,
                    sequential = false,
                ),
            )

            // Split-model routing: the DiT must NOT be in modelPath (else sdcpp loads it as a
            // complete checkpoint); it goes to diffusionModelPath and the encoder to llmPath.
            assertEquals(null, observedModelPath)
            assertEquals(ditFile.absolutePath, observedDiffusionModelPath)
            assertEquals(encFile.absolutePath, observedLlmPath)
            assertEquals(vaeFile.absolutePath, observedVaePath)
            // Split models offload to CPU to fit mobile memory budgets.
            assertEquals(true, observedOffload)
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
                any(),
                any(),
                any(),
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
                any(),
                any(),
                any(),
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
                any(),
                any(),
                any(),
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
    fun `image generation reuses the cached runtime across requests`() = runTest {
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
                any(),
                any(),
                any(),
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

            // Warm reuse makes the second request a cache hit with no second load.
            coVerify(exactly = 1) {
                StableDiffusion.loadWithRuntimeBackend(
                    any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
                    any(),
                    any(),
                    any(),
                )
            }
            assertFalse(metrics.first().cacheHit)
            assertTrue(metrics.first().modelLoadMs > 0L)
            assertTrue(metrics.first().runtimeAcquireMs >= metrics.first().modelLoadMs)
            assertTrue(metrics.last().cacheHit)
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
                any(),
                any(),
                any(),
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
                any(),
                any(),
                any(),
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
                any(),
                any(),
                any(),
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

    @Test
    fun `image generate with splitDiffusionModel and clip slots resolved componentPaths`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val ditFile = java.io.File.createTempFile("split-dit", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile = java.io.File.createTempFile("split-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val clipLFile = java.io.File.createTempFile("split-clip_l", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val clipGFile = java.io.File.createTempFile("split-clip_g", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        every { RuntimeCapabilities.isStableDiffusionVulkanAvailable() } returns true
        every { RuntimeCapabilities.isStableDiffusionOpenClAvailable() } returns false

        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
                override fun txt2img(
                    handle: Long, prompt: String, negative: String, width: Int, height: Int,
                    steps: Int, cfg: Float, seed: Long, vaeTiling: Boolean,
                    easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float, easyCacheEndPercent: Float,
                ): ByteArray = ByteArray(width * height * 3) { 0x33 }

                override fun txt2vid(
                    handle: Long, prompt: String, negative: String, width: Int, height: Int,
                    videoFrames: Int, steps: Int, cfg: Float, seed: Long,
                    sampleMethod: SampleMethod, scheduler: Scheduler, strength: Float,
                    initImage: ByteArray?, initWidth: Int, initHeight: Int,
                    vaceStrength: Float, easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float, easyCacheEndPercent: Float,
                ): Array<ByteArray> = Array(videoFrames) { ByteArray(width * height * 3) { 0x33 } }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit
                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        var capturedComponentPaths: io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths? = null
        var capturedLlmPath: String? = "unset"
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            capturedLlmPath = callArgs[22] as String?
            capturedComponentPaths = callArgs[23] as io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths?
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
            client.generate(
                ImageGenerationRequest(
                    prompt = "split model image",
                    width = 64,
                    height = 64,
                    model = ModelSpec.localFile(ditFile),
                    vae = ModelSpec.localFile(vaeFile),
                    clipL = ModelSpec.localFile(clipLFile),
                    clipG = ModelSpec.localFile(clipGFile),
                    splitDiffusionModel = true,
                ),
            )

            assertNotNull(capturedComponentPaths)
            assertEquals(clipLFile.absolutePath, capturedComponentPaths?.clipLPath)
            assertEquals(clipGFile.absolutePath, capturedComponentPaths?.clipGPath)
            assertEquals(null, capturedLlmPath)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `two requests identical except different clipG specs produce different cache keys and load twice`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile = java.io.File.createTempFile("flux-model-cache", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val clipG1 = java.io.File.createTempFile("clipG1", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val clipG2 = java.io.File.createTempFile("clipG2", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
                override fun txt2img(
                    handle: Long, prompt: String, negative: String, width: Int, height: Int,
                    steps: Int, cfg: Float, seed: Long, vaeTiling: Boolean,
                    easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float, easyCacheEndPercent: Float,
                ): ByteArray = ByteArray(width * height * 3) { 0x33 }

                override fun txt2vid(
                    handle: Long, prompt: String, negative: String, width: Int, height: Int,
                    videoFrames: Int, steps: Int, cfg: Float, seed: Long,
                    sampleMethod: SampleMethod, scheduler: Scheduler, strength: Float,
                    initImage: ByteArray?, initWidth: Int, initHeight: Int,
                    vaceStrength: Float, easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float, easyCacheEndPercent: Float,
                ): Array<ByteArray>? = null

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit
                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
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
            client.generate(
                ImageGenerationRequest(
                    prompt = "cache key test 1",
                    width = 64,
                    height = 64,
                    model = ModelSpec.localFile(modelFile),
                    clipG = ModelSpec.localFile(clipG1),
                ),
            )

            client.generate(
                ImageGenerationRequest(
                    prompt = "cache key test 2",
                    width = 64,
                    height = 64,
                    model = ModelSpec.localFile(modelFile),
                    clipG = ModelSpec.localFile(clipG2),
                ),
            )

            // Should load twice because the cache keys are different due to different clipG specs
            coVerify(exactly = 2) {
                StableDiffusion.loadWithRuntimeBackend(
                    any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
                    any(),
                    any(),
                    any(),
                )
            }
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `t5xxl model spec resolves and routes to native t5xxlPath while textEncoder routes to llmPath for split model`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val ditFile =
            java.io.File.createTempFile("flux2-dit", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile =
            java.io.File.createTempFile("flux2-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val encFile =
            java.io.File.createTempFile("qwen3-enc", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5xxlFile =
            java.io.File.createTempFile("t5xxl", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

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
                ): Array<ByteArray> = Array(1) { ByteArray(width * height * 3) { 0x33 } }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        var observedModelPath: String? = "unset"
        var observedDiffusionModelPath: String? = "unset"
        var observedLlmPath: String? = "unset"
        var observedT5xxlPath: String? = "unset"
        var observedVaePath: String? = "unset"
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            observedModelPath = callArgs[3] as String?
            observedVaePath = callArgs[4] as String?
            observedT5xxlPath = callArgs[5] as String?
            observedDiffusionModelPath = callArgs[21] as String?
            observedLlmPath = callArgs[22] as String?
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
            client.generate(
                ImageGenerationRequest(
                    prompt = "test t5xxl split routing",
                    width = 256,
                    height = 256,
                    model = ModelSpec.localFile(ditFile),
                    vae = ModelSpec.localFile(vaeFile),
                    textEncoder = ModelSpec.localFile(encFile),
                    t5xxl = ModelSpec.localFile(t5xxlFile),
                    splitDiffusionModel = true,
                    sequential = false,
                ),
            )

            assertEquals(null, observedModelPath)
            assertEquals(ditFile.absolutePath, observedDiffusionModelPath)
            assertEquals(encFile.absolutePath, observedLlmPath)
            assertEquals(t5xxlFile.absolutePath, observedT5xxlPath)
            assertEquals(vaeFile.absolutePath, observedVaePath)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `two requests identical except different t5xxl specs produce different cache keys and load twice`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val modelFile = java.io.File.createTempFile("flux-model-cache", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5xxl1 = java.io.File.createTempFile("t5xxl1", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5xxl2 = java.io.File.createTempFile("t5xxl2", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
                override fun txt2img(
                    handle: Long, prompt: String, negative: String, width: Int, height: Int,
                    steps: Int, cfg: Float, seed: Long, vaeTiling: Boolean,
                    easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float, easyCacheEndPercent: Float,
                ): ByteArray = ByteArray(width * height * 3) { 0x33 }

                override fun txt2vid(
                    handle: Long, prompt: String, negative: String, width: Int, height: Int,
                    videoFrames: Int, steps: Int, cfg: Float, seed: Long,
                    sampleMethod: SampleMethod, scheduler: Scheduler, strength: Float,
                    initImage: ByteArray?, initWidth: Int, initHeight: Int,
                    vaceStrength: Float, easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float, easyCacheEndPercent: Float,
                ): Array<ByteArray>? = null

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit
                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
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
            client.generate(
                ImageGenerationRequest(
                    prompt = "cache key test 1",
                    width = 64,
                    height = 64,
                    model = ModelSpec.localFile(modelFile),
                    t5xxl = ModelSpec.localFile(t5xxl1),
                ),
            )

            client.generate(
                ImageGenerationRequest(
                    prompt = "cache key test 2",
                    width = 64,
                    height = 64,
                    model = ModelSpec.localFile(modelFile),
                    t5xxl = ModelSpec.localFile(t5xxl2),
                ),
            )

            // Should load twice because the cache keys are different due to different t5xxl specs
            coVerify(exactly = 2) {
                StableDiffusion.loadWithRuntimeBackend(
                    any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
                    any(),
                    any(),
                    any(),
                )
            }
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `SD3 CLIP-only encoderOnly spec resolves and routes to native with null modelPath, null vae, null t5xxl, null diffusion, null llm, and populated clipL and clipG componentPaths`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val clipLFile = java.io.File.createTempFile("clip_l", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val clipGFile = java.io.File.createTempFile("clip_g", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        var observedModelPath: String? = "unset"
        var observedVaePath: String? = "unset"
        var observedT5xxlPath: String? = "unset"
        var observedDiffusionModelPath: String? = "unset"
        var observedLlmPath: String? = "unset"
        var observedComponentPaths: io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths? = null

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            observedModelPath = callArgs[3] as String?
            observedVaePath = callArgs[4] as String?
            observedT5xxlPath = callArgs[5] as String?
            observedDiffusionModelPath = callArgs[21] as String?
            observedLlmPath = callArgs[22] as String?
            observedComponentPaths = callArgs[23] as io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths?
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val spec = DiffusionRuntimeSpec(
            role = DiffusionRuntimeRole.IMAGE,
            model = ModelSpec.localFile(clipLFile),
            t5xxl = null,
            clipL = ModelSpec.localFile(clipLFile),
            clipG = ModelSpec.localFile(clipGFile),
            encoderOnly = true
        )

        val loader = DiffusionRuntimeLoader(context, DefaultModelRepository())
        val loaded = loader.load(
            spec = spec,
            options = DiffusionLoadOptions(
                subsystem = ComputeSubsystem.IMAGE,
                allowGpu = false,
                nThreads = 1,
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                preferPerformanceMode = false,
            ),
            backend = ComputeBackend.CPU
        )
        try {
            assertEquals(null, observedModelPath)
            assertEquals(null, observedVaePath)
            assertEquals(null, observedT5xxlPath)
            assertEquals(null, observedDiffusionModelPath)
            assertEquals(null, observedLlmPath)
            assertNotNull(observedComponentPaths)
            assertEquals(clipLFile.absolutePath, observedComponentPaths?.clipLPath)
            assertEquals(clipGFile.absolutePath, observedComponentPaths?.clipGPath)
        } finally {
            loaded.close()
        }
    }

    @Test
    fun `SD3 T5-only encoderOnly spec resolves and routes to native with null modelPath, null vae, populated t5xxl, null diffusion, null llm, and null componentPaths`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val t5xxlFile = java.io.File.createTempFile("t5xxl", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        var observedModelPath: String? = "unset"
        var observedVaePath: String? = "unset"
        var observedT5xxlPath: String? = "unset"
        var observedDiffusionModelPath: String? = "unset"
        var observedLlmPath: String? = "unset"
        var observedComponentPaths: io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths? = null

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            observedModelPath = callArgs[3] as String?
            observedVaePath = callArgs[4] as String?
            observedT5xxlPath = callArgs[5] as String?
            observedDiffusionModelPath = callArgs[21] as String?
            observedLlmPath = callArgs[22] as String?
            observedComponentPaths = callArgs[23] as io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths?
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val spec = DiffusionRuntimeSpec(
            role = DiffusionRuntimeRole.IMAGE,
            model = ModelSpec.localFile(t5xxlFile),
            t5xxl = ModelSpec.localFile(t5xxlFile),
            clipL = null,
            clipG = null,
            encoderOnly = true
        )

        val loader = DiffusionRuntimeLoader(context, DefaultModelRepository())
        val loaded = loader.load(
            spec = spec,
            options = DiffusionLoadOptions(
                subsystem = ComputeSubsystem.IMAGE,
                allowGpu = false,
                nThreads = 1,
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                preferPerformanceMode = false,
            ),
            backend = ComputeBackend.CPU
        )
        try {
            assertEquals(null, observedModelPath)
            assertEquals(null, observedVaePath)
            assertEquals(t5xxlFile.absolutePath, observedT5xxlPath)
            assertEquals(null, observedDiffusionModelPath)
            assertEquals(null, observedLlmPath)
            assertEquals(null, observedComponentPaths)
        } finally {
            loaded.close()
        }
    }

    @Test
    fun `masked T5 encoderOnly spec routes MiniT2I conditioning through modelPath`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val encoderFile = java.io.File.createTempFile("flan-t5", ".safetensors", context.filesDir).apply { writeBytes(byteArrayOf(0x01)) }
        var observedModelPath: String? = "unset"
        var observedVaePath: String? = "unset"
        var observedT5xxlPath: String? = "unset"
        var observedDiffusionModelPath: String? = "unset"
        var observedLlmPath: String? = "unset"
        var observedComponentPaths: io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths? = null

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            observedModelPath = callArgs[3] as String?
            observedVaePath = callArgs[4] as String?
            observedT5xxlPath = callArgs[5] as String?
            observedDiffusionModelPath = callArgs[21] as String?
            observedLlmPath = callArgs[22] as String?
            observedComponentPaths = callArgs[23] as io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths?
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val loaded =
            DiffusionRuntimeLoader(context, DefaultModelRepository()).load(
                spec =
                    DiffusionRuntimeSpec(
                        role = DiffusionRuntimeRole.IMAGE,
                        model = ModelSpec.localFile(encoderFile),
                        encoderOnly = true,
                        conditioningProfile = ImageConditioningProfile.MASKED_T5,
                    ),
                options =
                    DiffusionLoadOptions(
                        subsystem = ComputeSubsystem.IMAGE,
                        allowGpu = false,
                        nThreads = 1,
                        offloadToCpu = true,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = true,
                        preferPerformanceMode = false,
                    ),
                backend = ComputeBackend.CPU,
            )
        try {
            assertEquals(encoderFile.absolutePath, observedModelPath)
            assertEquals(null, observedVaePath)
            assertEquals(null, observedT5xxlPath)
            assertEquals(null, observedDiffusionModelPath)
            assertEquals(null, observedLlmPath)
            assertEquals(true, observedComponentPaths?.miniT2iConditionerOnly)
        } finally {
            loaded.close()
        }
    }

    @Test
    fun `StableDiffusionLoader load propagates weightType and tensorTypeRules`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val dummyFile = java.io.File.createTempFile("dummy", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        var observedRequest: io.aatricks.llmedge.image.diffusion.StableDiffusionNativeLoadRequest? = null

        mockkObject(StableDiffusion.Companion)
        every {
            StableDiffusion.Companion.supportNativeCreate(any())
        } answers {
            observedRequest = it.invocation.args[0] as io.aatricks.llmedge.image.diffusion.StableDiffusionNativeLoadRequest
            1L
        }

        try {
            val loadRequest = StableDiffusionLoadRequest(
                assets = StableDiffusionAssetRequest(
                    modelPath = dummyFile.absolutePath,
                    componentPaths = StableDiffusionComponentPaths(
                        weightType = "q8_0",
                        tensorTypeRules = ".*mask_token.*=f16",
                    ),
                ),
                runtime = StableDiffusionRuntimeRequest(
                    nThreads = 1,
                    offloadToCpu = false,
                    keepClipOnCpu = false,
                    keepVaeOnCpu = false,
                    flashAttn = true,
                    vaeDecodeOnly = true,
                    sequentialLoad = false,
                    preferPerformanceMode = false,
                    flowShift = Float.POSITIVE_INFINITY,
                    loraApplyMode = LoraApplyMode.AUTO,
                ),
                backend = StableDiffusionBackendRequest(
                    allowOpenCl = false,
                    allowVulkan = false,
                    forceVulkan = false,
                    allowBackendFallbackToCpu = true,
                ),
            )

            val loaded = StableDiffusionLoader.load(context, loadRequest)
            loaded.close()

            assertNotNull(observedRequest)
            assertEquals("q8_0", observedRequest?.weightType)
            assertEquals(".*mask_token.*=f16", observedRequest?.tensorTypeRules)
        } finally {
            dummyFile.delete()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `pure condition combination yields the expected arrays and dims and rejects mismatches`() {
        val condA = PrecomputedCondition(
            cCrossAttn = floatArrayOf(1.0f, 2.0f),
            cCrossAttnDims = intArrayOf(2, 1),
            cVector = floatArrayOf(5.0f),
            cVectorDims = intArrayOf(1),
            cConcat = null,
            cConcatDims = null
        )
        val condB = PrecomputedCondition(
            cCrossAttn = floatArrayOf(10.0f, 20.0f),
            cCrossAttnDims = intArrayOf(2, 1),
            cVector = floatArrayOf(50.0f),
            cVectorDims = intArrayOf(1, 1),
            cConcat = null,
            cConcatDims = null
        )

        val executorClass = io.aatricks.llmedge.image.ImageGenerationExecutor::class.java
        val method = executorClass.getDeclaredMethod("combineSD3Condition", PrecomputedCondition::class.java, PrecomputedCondition::class.java)
        method.isAccessible = true

        val dummyExecutor = io.aatricks.llmedge.image.ImageGenerationExecutor(
            config = io.aatricks.llmedge.LLMEdgeConfig(),
            generationMutex = kotlinx.coroutines.sync.Mutex(),
            imageRequestIds = java.util.concurrent.atomic.AtomicLong(0),
            state = io.aatricks.llmedge.image.ImageClientState(),
            requestExecutor = mockk(relaxed = true),
            executionPlanSelector = ImageExecutionPlanSelector { _, _ -> error("not used") },
            logTag = "TestExecutor"
        )

        val result = method.invoke(dummyExecutor, condA, condB) as PrecomputedCondition
        org.junit.Assert.assertArrayEquals(floatArrayOf(11.0f, 22.0f), result.cCrossAttn, 1e-5f)
        org.junit.Assert.assertArrayEquals(intArrayOf(2, 1), result.cCrossAttnDims)
        org.junit.Assert.assertArrayEquals(floatArrayOf(55.0f), result.cVector, 1e-5f)
        org.junit.Assert.assertArrayEquals(intArrayOf(1, 1), result.cVectorDims)

        val condMissingCrossAttn = PrecomputedCondition(
            cCrossAttn = null,
            cCrossAttnDims = null,
            cVector = floatArrayOf(5.0f),
            cVectorDims = intArrayOf(1, 1)
        )
        try {
            method.invoke(dummyExecutor, condA, condMissingCrossAttn)
            org.junit.Assert.fail("Expected exception when cCrossAttn is missing on one side")
        } catch (e: java.lang.reflect.InvocationTargetException) {
            assertTrue(e.targetException is IllegalArgumentException)
        }

        val condBadDims = PrecomputedCondition(
            cCrossAttn = floatArrayOf(10.0f, 20.0f),
            cCrossAttnDims = intArrayOf(1, 2),
            cVector = floatArrayOf(50.0f),
            cVectorDims = intArrayOf(1, 1)
        )
        try {
            method.invoke(dummyExecutor, condA, condBadDims)
            org.junit.Assert.fail("Expected exception when cCrossAttn dimensions mismatch")
        } catch (e: java.lang.reflect.InvocationTargetException) {
            assertTrue(e.targetException is IllegalArgumentException)
        }

        val condBadSize = PrecomputedCondition(
            cCrossAttn = floatArrayOf(10.0f),
            cCrossAttnDims = intArrayOf(2, 1),
            cVector = floatArrayOf(50.0f),
            cVectorDims = intArrayOf(1, 1)
        )
        try {
            method.invoke(dummyExecutor, condA, condBadSize)
            org.junit.Assert.fail("Expected exception when cCrossAttn sizes mismatch")
        } catch (e: java.lang.reflect.InvocationTargetException) {
            assertTrue(e.targetException is IllegalArgumentException)
        }
    }

    @Test
    fun `SD3 sequential generation loads CLIP and T5 sub-phases separately, precomputes prompt and negative in each, invalidates runtimes in between, and passes combined cond and uncond to diffusion`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val ditFile = java.io.File.createTempFile("sd3-dit", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile = java.io.File.createTempFile("sd3-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val clipLFile = java.io.File.createTempFile("sd3-clipL", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val clipGFile = java.io.File.createTempFile("sd3-clipG", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5xxlFile = java.io.File.createTempFile("sd3-t5xxl", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        val events = mutableListOf<String>()
        var observedCond: PrecomputedCondition? = null
        var observedUncond: PrecomputedCondition? = null
        var clipModelClosedBeforeT5Load = false
        var t5ModelClosedBeforeDitLoad = false
        var clipModelClosed = false
        var t5ModelClosed = false

        val clipModel = mockk<StableDiffusion>(relaxed = true)
        val t5Model = mockk<StableDiffusion>(relaxed = true)
        val ditModel = mockk<StableDiffusion>(relaxed = true)

        val condCLIP = PrecomputedCondition(
            cCrossAttn = floatArrayOf(1.0f),
            cCrossAttnDims = intArrayOf(1, 1),
            cVector = floatArrayOf(10.0f),
            cVectorDims = intArrayOf(1, 1),
        )
        val uncondCLIP = PrecomputedCondition(
            cCrossAttn = floatArrayOf(2.0f),
            cCrossAttnDims = intArrayOf(1, 1),
            cVector = floatArrayOf(20.0f),
            cVectorDims = intArrayOf(1, 1),
        )
        val condT5 = PrecomputedCondition(
            cCrossAttn = floatArrayOf(3.0f),
            cCrossAttnDims = intArrayOf(1, 1),
            cVector = floatArrayOf(30.0f),
            cVectorDims = intArrayOf(1, 1),
        )
        val uncondT5 = PrecomputedCondition(
            cCrossAttn = floatArrayOf(4.0f),
            cCrossAttnDims = intArrayOf(1, 1),
            cVector = floatArrayOf(40.0f),
            cVectorDims = intArrayOf(1, 1),
        )

        coEvery { clipModel.precomputeCondition(any(), any(), any(), any(), any()) } coAnswers {
            val promptArg = arg<String>(0)
            events.add("clip.precompute($promptArg)")
            if (promptArg == "test prompt") condCLIP else uncondCLIP
        }

        coEvery { t5Model.precomputeCondition(any(), any(), any(), any(), any()) } coAnswers {
            val promptArg = arg<String>(0)
            events.add("t5.precompute($promptArg)")
            if (promptArg == "test prompt") condT5 else uncondT5
        }

        every { clipModel.close() } answers {
            clipModelClosed = true
            events.add("clip.close()")
        }

        every { t5Model.close() } answers {
            t5ModelClosed = true
            events.add("t5.close()")
        }

        every { ditModel.txt2ImgWithPrecomputedCondition(
            any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any()
        ) } answers {
            val callArgs = it.invocation.args
            observedCond = callArgs[8] as PrecomputedCondition?
            observedUncond = callArgs[9] as PrecomputedCondition?
            events.add("txt2ImgWithPrecomputedCondition")
            ByteArray(256 * 256 * 3) { 0 }
        }

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            val isClip = (callArgs[23] as? io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths)?.clipLPath != null
            val isT5 = callArgs[5] as? String? != null
            if (isClip) {
                events.add("load(CLIP)")
                clipModel
            } else if (isT5) {
                events.add("load(T5)")
                if (clipModelClosed) {
                    clipModelClosedBeforeT5Load = true
                }
                t5Model
            } else {
                events.add("load(diffusion)")
                if (t5ModelClosed) {
                    t5ModelClosedBeforeDitLoad = true
                }
                ditModel
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
            client.generate(
                ImageGenerationRequest(
                    prompt = "test prompt",
                    negative = "test negative",
                    width = 256,
                    height = 256,
                    sequential = true,
                    splitDiffusionModel = true,
                    model = ModelSpec.localFile(ditFile),
                    vae = ModelSpec.localFile(vaeFile),
                    clipL = ModelSpec.localFile(clipLFile),
                    clipG = ModelSpec.localFile(clipGFile),
                    t5xxl = ModelSpec.localFile(t5xxlFile),
                ),
            )

            val expectedEvents = listOf(
                "load(CLIP)",
                "clip.precompute(test prompt)",
                "clip.precompute(test negative)",
                "clip.close()",
                "load(T5)",
                "t5.precompute(test prompt)",
                "t5.precompute(test negative)",
                "t5.close()",
                "load(diffusion)",
                "txt2ImgWithPrecomputedCondition"
            )
            assertEquals(expectedEvents, events)
            assertTrue("CLIP model must be closed before T5 model is loaded", clipModelClosedBeforeT5Load)
            assertTrue("T5 model must be closed before DiT model is loaded", t5ModelClosedBeforeDitLoad)

            assertNotNull(observedCond)
            org.junit.Assert.assertArrayEquals(floatArrayOf(4.0f), observedCond!!.cCrossAttn, 1e-5f)
            org.junit.Assert.assertArrayEquals(floatArrayOf(40.0f), observedCond!!.cVector, 1e-5f)

            assertNotNull(observedUncond)
            org.junit.Assert.assertArrayEquals(floatArrayOf(6.0f), observedUncond!!.cCrossAttn, 1e-5f)
            org.junit.Assert.assertArrayEquals(floatArrayOf(60.0f), observedUncond!!.cVector, 1e-5f)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `FLUX sequential generation precomputes prompt and invalidates conditioning before diffusion`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val ditFile = java.io.File.createTempFile("flux-dit", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val vaeFile = java.io.File.createTempFile("flux-vae", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val encFile = java.io.File.createTempFile("flux-enc", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        val events = mutableListOf<String>()
        var observedCond: PrecomputedCondition? = null
        var observedUncond: PrecomputedCondition? = null
        var condModelClosedBeforeDitLoad = false
        var condModelClosed = false

        val condModel = mockk<StableDiffusion>(relaxed = true)
        val ditModel = mockk<StableDiffusion>(relaxed = true)

        val condResult = PrecomputedCondition(
            cCrossAttn = floatArrayOf(3.0f),
            cCrossAttnDims = intArrayOf(1, 1),
            cVector = floatArrayOf(3.0f),
            cVectorDims = intArrayOf(1, 1),
            cConcat = floatArrayOf(3.0f),
            cConcatDims = intArrayOf(1, 1),
        )

        coEvery { condModel.precomputeCondition(any(), any(), any(), any(), any()) } coAnswers {
            val promptArg = arg<String>(0)
            events.add("precomputeCondition($promptArg)")
            condResult
        }

        every { condModel.close() } answers {
            condModelClosed = true
            events.add("condModel.close()")
        }

        every { ditModel.txt2ImgWithPrecomputedCondition(
            any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any()
        ) } answers {
            val callArgs = it.invocation.args
            observedCond = callArgs[8] as PrecomputedCondition?
            observedUncond = callArgs[9] as PrecomputedCondition?
            events.add("txt2ImgWithPrecomputedCondition")
            ByteArray(256 * 256 * 3) { 0 }
        }

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            val isCond = callArgs[22] as String? != null
            if (isCond) {
                events.add("load(conditioning)")
                condModel
            } else {
                events.add("load(diffusion)")
                if (condModelClosed) {
                    condModelClosedBeforeDitLoad = true
                }
                ditModel
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
            client.generate(
                ImageGenerationRequest(
                    prompt = "test prompt",
                    width = 256,
                    height = 256,
                    sequential = true,
                    splitDiffusionModel = true,
                    model = ModelSpec.localFile(ditFile),
                    vae = ModelSpec.localFile(vaeFile),
                    textEncoder = ModelSpec.localFile(encFile),
                ),
            )

            val expectedEvents = listOf(
                "load(conditioning)",
                "precomputeCondition(test prompt)",
                "condModel.close()",
                "load(diffusion)",
                "txt2ImgWithPrecomputedCondition"
            )
            assertEquals(expectedEvents, events)
            assertTrue("Conditioning model must be closed before DiT model is loaded", condModelClosedBeforeDitLoad)
            org.junit.Assert.assertSame(condResult, observedCond)
            assertEquals(null, observedUncond)
        } finally {
            client.close()
            edgeScope.close()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `automatic direct load failure retries once with the staged flux plan`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val ditFile = java.io.File.createTempFile("flux-dit", ".gguf", context.filesDir).apply { writeBytes(byteArrayOf(0x01)) }
        val encoderFile = java.io.File.createTempFile("flux-encoder", ".gguf", context.filesDir).apply { writeBytes(byteArrayOf(0x01)) }
        val loads = mutableListOf<String>()
        val conditioningModel = mockk<StableDiffusion>(relaxed = true)
        val diffusionModel = mockk<StableDiffusion>(relaxed = true)
        val condition =
            PrecomputedCondition(
                cCrossAttn = floatArrayOf(1f),
                cCrossAttnDims = intArrayOf(1, 1),
                cVector = floatArrayOf(1f),
                cVectorDims = intArrayOf(1, 1),
            )

        coEvery { conditioningModel.precomputeCondition(any(), any(), any(), any(), any()) } returns condition
        every {
            diffusionModel.txt2ImgWithPrecomputedCondition(
                any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(),
            )
        } returns ByteArray(256 * 256 * 3)
        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(),
            )
        } coAnswers {
            val diffusionPath = it.invocation.args[21] as String?
            val llmPath = it.invocation.args[22] as String?
            when {
                diffusionPath != null && llmPath != null -> {
                    loads += "direct"
                    throw ModelLoadException(diffusionPath, "synthetic direct-load failure")
                }
                diffusionPath == null && llmPath != null -> {
                    loads += "conditioning"
                    conditioningModel
                }
                diffusionPath != null -> {
                    loads += "diffusion"
                    diffusionModel
                }
                else -> error("Unexpected FLUX runtime load")
            }
        }

        val config = LLMEdgeConfig(image = ImageRuntimeConfig(useVulkan = false))
        val scope = LLMEdgeScope(this, 1)
        val runtimePool = createDiffusionRuntimePool(context, scope, config, DefaultModelRepository())
        val state = ImageClientState()
        val executor =
            ImageGenerationExecutor(
                config = config,
                generationMutex = kotlinx.coroutines.sync.Mutex(),
                imageRequestIds = java.util.concurrent.atomic.AtomicLong(0),
                state = state,
                requestExecutor = DiffusionRequestExecutor(runtimePool, state, "TestExecutor"),
                executionPlanSelector =
                    ImageExecutionPlanSelector { params, _ ->
                        ImageExecutionDecision(
                            mode = ImageExecutionMode.DIRECT,
                            reason = "TEST_DIRECT",
                            recipe = ImageExecutionPlanner.recipeFor(params),
                        )
                    },
                logTag = "TestExecutor",
            )

        try {
            executor.generate(
                ImageGenerationRequest(
                    prompt = "test prompt",
                    width = 256,
                    height = 256,
                    cfgScale = 1f,
                    model = ModelSpec.localFile(ditFile),
                    textEncoder = ModelSpec.localFile(encoderFile),
                    splitDiffusionModel = true,
                ),
            )

            assertEquals(listOf("direct", "direct", "conditioning", "diffusion"), loads)
        } finally {
            runtimePool.close()
            scope.close()
        }
    }

    @Test
    fun `createDiffusionRuntimePool cache key differentiates quantized and default runtimes`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val dummyFile = java.io.File(context.filesDir, "dummy_model.safetensors").apply { writeBytes(byteArrayOf(0x01)) }
        val fakeRepo = object : io.aatricks.llmedge.model.ModelRepository {
            override suspend fun resolve(
                context: Context,
                spec: ModelSpec,
                onProgress: ((io.aatricks.llmedge.core.ProgressEvent.Downloading) -> Unit)?,
            ): java.io.File = dummyFile
        }

        mockkObject(StableDiffusion.Companion)
        val capturedRequests = mutableListOf<StableDiffusionNativeLoadRequest>()
        every {
            StableDiffusion.Companion.supportNativeCreate(capture(capturedRequests))
        } returns 1L

        val config = LLMEdgeConfig()
        val scope = LLMEdgeScope(this, 1)
        val runtimePool = createDiffusionRuntimePool(context, scope, config, fakeRepo)

        try {
            val largeRequestDirect = ImageRuntimeRequestPlanner.imageRequest(MiniT2I.largeImageRequest("x"), config)
            val standardRequestDirect = ImageRuntimeRequestPlanner.imageRequest(MiniT2I.imageRequest("x"), config)

            val resultLarge = runtimePool.coordinator.acquireDetailed(largeRequestDirect.spec, largeRequestDirect.options)
            val resultStandard = runtimePool.coordinator.acquireDetailed(standardRequestDirect.spec, standardRequestDirect.options)

            assertTrue(resultLarge.keyPrefix.contains("weightType=q8_0"))
            assertTrue(resultLarge.keyPrefix.contains("tensorTypeRules=.*mask_token.*=f16"))

            assertTrue(resultStandard.keyPrefix.contains("weightType=null"))
            assertTrue(resultStandard.keyPrefix.contains("tensorTypeRules=null"))

            org.junit.Assert.assertNotEquals(resultLarge.keyPrefix, resultStandard.keyPrefix)

            val largeReq = capturedRequests.find { it.weightType == "q8_0" }
            val standardReq = capturedRequests.find { it.weightType == null }

            org.junit.Assert.assertNotNull(largeReq)
            org.junit.Assert.assertNotNull(standardReq)
            assertEquals(".*mask_token.*=f16", largeReq?.tensorTypeRules)
            org.junit.Assert.assertNull(standardReq?.tensorTypeRules)
        } finally {
            runtimePool.close()
            scope.close()
            dummyFile.delete()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `Chroma sequential image execution precomputes positive and negative conditions and passes both to diffusion`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val baseDir = context.filesDir
        val ditFile = java.io.File.createTempFile("chroma-dit", ".gguf", baseDir).apply { writeBytes(byteArrayOf(0x01)) }
        val t5File = java.io.File.createTempFile("chroma-t5", ".safetensors", baseDir).apply { writeBytes(byteArrayOf(0x01)) }

        val events = mutableListOf<String>()
        var observedCond: PrecomputedCondition? = null
        var observedUncond: PrecomputedCondition? = null
        var t5ModelClosedBeforeDitLoad = false
        var t5ModelClosed = false

        val t5Model = mockk<StableDiffusion>(relaxed = true)
        val ditModel = mockk<StableDiffusion>(relaxed = true)

        val condChroma = PrecomputedCondition(
            cCrossAttn = floatArrayOf(5.0f),
            cCrossAttnDims = intArrayOf(1, 1),
        )
        val uncondChroma = PrecomputedCondition(
            cCrossAttn = floatArrayOf(6.0f),
            cCrossAttnDims = intArrayOf(1, 1),
        )

        coEvery { t5Model.precomputeCondition(any(), any(), any(), any(), any()) } coAnswers {
            val promptArg = arg<String>(0)
            events.add("t5.precompute($promptArg)")
            if (promptArg == "test prompt") condChroma else uncondChroma
        }

        every { t5Model.close() } answers {
            t5ModelClosed = true
            events.add("t5.close()")
        }

        every { ditModel.txt2ImgWithPrecomputedCondition(
            any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any()
        ) } answers {
            val callArgs = it.invocation.args
            observedCond = callArgs[8] as PrecomputedCondition?
            observedUncond = callArgs[9] as PrecomputedCondition?
            events.add("txt2ImgWithPrecomputedCondition")
            ByteArray(256 * 256 * 3) { 0 }
        }

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            val isT5 = callArgs[5] as? String? != null
            if (isT5) {
                events.add("load(T5)")
                t5Model
            } else {
                events.add("load(diffusion)")
                if (t5ModelClosed) {
                    t5ModelClosedBeforeDitLoad = true
                }
                ditModel
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
            client.generate(
                ImageGenerationRequest(
                    prompt = "test prompt",
                    width = 256,
                    height = 256,
                    sequential = true,
                    splitDiffusionModel = true,
                    model = ModelSpec.localFile(ditFile),
                    t5xxl = ModelSpec.localFile(t5File),
                ),
            )

            val expectedEvents = listOf(
                "load(T5)",
                "t5.precompute(test prompt)",
                "t5.precompute()",
                "t5.close()",
                "load(diffusion)",
                "txt2ImgWithPrecomputedCondition"
            )
            assertEquals(expectedEvents, events)
            assertTrue("T5 encoder must be closed before DiT model is loaded", t5ModelClosedBeforeDitLoad)
            org.junit.Assert.assertSame(condChroma, observedCond)
            org.junit.Assert.assertSame(uncondChroma, observedUncond)
        } finally {
            client.close()
            edgeScope.close()
            ditFile.delete()
            t5File.delete()
            StableDiffusion.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `chroma T5 encoderOnly spec requires chromaT5ConditionerOnly signal keeping T5 in t5xxlPath and specifying Chroma mask_pad=1`(
    ) = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val encoderFile =
            java.io.File.createTempFile("chroma-t5", ".safetensors", context.filesDir).apply {
                writeBytes(byteArrayOf(0x01))
            }
        var observedModelPath: String? = "unset"
        var observedVaePath: String? = "unset"
        var observedT5xxlPath: String? = "unset"
        var observedDiffusionModelPath: String? = "unset"
        var observedLlmPath: String? = "unset"
        var observedComponentPaths: io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths? = null

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(),
                any(),
                any(),
            )
        } coAnswers {
            val callArgs = it.invocation.args
            observedModelPath = callArgs[3] as String?
            observedVaePath = callArgs[4] as String?
            observedT5xxlPath = callArgs[5] as String?
            observedDiffusionModelPath = callArgs[21] as String?
            observedLlmPath = callArgs[22] as String?
            observedComponentPaths = callArgs[23] as io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths?
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val loaded =
            DiffusionRuntimeLoader(context, DefaultModelRepository()).load(
                spec =
                    DiffusionRuntimeSpec(
                        role = DiffusionRuntimeRole.IMAGE,
                        model = ModelSpec.localFile(encoderFile),
                        encoderOnly = true,
                        conditioningProfile = ImageConditioningProfile.CHROMA_T5,
                    ),
                options =
                    DiffusionLoadOptions(
                        subsystem = ComputeSubsystem.IMAGE,
                        allowGpu = false,
                        nThreads = 1,
                        offloadToCpu = true,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = true,
                        preferPerformanceMode = false,
                    ),
                backend = ComputeBackend.CPU,
            )
        try {
            assertEquals(null, observedModelPath)
            assertEquals(encoderFile.absolutePath, observedT5xxlPath)
            assertEquals(null, observedVaePath)
            assertEquals(null, observedDiffusionModelPath)
            assertEquals(null, observedLlmPath)
            assertEquals(true, observedComponentPaths?.chromaT5ConditionerOnly)
        } finally {
            loaded.close()
        }
    }
}
