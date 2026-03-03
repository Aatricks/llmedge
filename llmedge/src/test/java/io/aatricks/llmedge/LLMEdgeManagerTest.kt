package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockkObject
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.junit.runner.RunWith

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class LLMEdgeManagerTest {

    @Before
    fun setup() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        // Mock the companion object so we can intercept calls to StableDiffusion.load
        mockkObject(StableDiffusion.Companion)
    }

    @After
    fun teardown() {
        StableDiffusion.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
        try { io.mockk.unmockkObject(StableDiffusion.Companion) } catch (_: Throwable) {}
        clearAllMocks()
    }

    @Test
    fun `generateVideoSequentially should load T5 for precompute then load diffusion without T5`() = runTest {
        // Setup mock environment
        val context = ApplicationProvider.getApplicationContext<Context>()

        // Create dummy files to be returned by HuggingFaceHub.ensureRepoFileOnDisk
        val baseDir = context.filesDir
        val modelFile = java.io.File.createTempFile("wan-model", ".gguf", baseDir)
        val vaeFile = java.io.File.createTempFile("wan-vae", ".safetensors", baseDir)
        val t5File = java.io.File.createTempFile("umt5", ".gguf", baseDir)

        // Ensure stubbed HF calls return our files
        mockkObject(HuggingFaceHub)
        coEvery {
            HuggingFaceHub.ensureRepoFileOnDisk(context, "Comfy-Org/Wan_2.1_ComfyUI_repackaged", any(), "wan2.1_t2v_1.3B_fp16.safetensors", any(), any(), any(), any(), any())
        } returns HuggingFaceHub.ModelDownloadResult(
            requestedModelId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            requestedRevision = "main",
            modelId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            revision = "main",
            file = modelFile,
            fileInfo = HuggingFaceHub.ModelFileMetadata(path = modelFile.name, sizeBytes = modelFile.length(), sha256 = null),
            fromCache = true,
            aliasApplied = false
        )

        coEvery {
            HuggingFaceHub.ensureRepoFileOnDisk(context, "Comfy-Org/Wan_2.1_ComfyUI_repackaged", any(), "wan_2.1_vae.safetensors", any(), any(), any(), any(), any())
        } returns HuggingFaceHub.ModelDownloadResult(
            requestedModelId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            requestedRevision = "main",
            modelId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            revision = "main",
            file = vaeFile,
            fileInfo = HuggingFaceHub.ModelFileMetadata(path = vaeFile.name, sizeBytes = vaeFile.length(), sha256 = null),
            fromCache = true,
            aliasApplied = false
        )

        coEvery {
            HuggingFaceHub.ensureRepoFileOnDisk(context, "city96/umt5-xxl-encoder-gguf", any(), "umt5-xxl-encoder-Q3_K_S.gguf", any(), any(), any(), any(), any())
        } returns HuggingFaceHub.ModelDownloadResult(
            requestedModelId = "city96/umt5-xxl-encoder-gguf",
            requestedRevision = "main",
            modelId = "city96/umt5-xxl-encoder-gguf",
            revision = "main",
            file = t5File,
            fileInfo = HuggingFaceHub.ModelFileMetadata(path = t5File.name, sizeBytes = t5File.length(), sha256 = null),
            fromCache = true,
            aliasApplied = false
        )

        // Override StableDiffusion native bridge so precompute and generation can be mocked
        StableDiffusion.overrideNativeBridgeForTests { instance ->
            object : StableDiffusion.NativeBridge {
                override fun txt2img(handle: Long, prompt: String, negative: String, width: Int, height: Int, steps: Int, cfg: Float, seed: Long, vaeTiling: Boolean, easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float, easyCacheStartPercent: Float, easyCacheEndPercent: Float): ByteArray? {
                    return ByteArray(3 * width * height) { 0 }
                }

                override fun txt2vid(handle: Long, prompt: String, negative: String, width: Int, height: Int,
                                     videoFrames: Int, steps: Int, cfg: Float, seed: Long, sampleMethod: StableDiffusion.SampleMethod, scheduler: StableDiffusion.Scheduler,
                                     strength: Float, initImage: ByteArray?, initWidth: Int, initHeight: Int, vaceStrength: Float, easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float, easyCacheStartPercent: Float, easyCacheEndPercent: Float): Array<ByteArray>? {
                    return Array(videoFrames) { ByteArray(width * height * 3) { ((it + 1) % 255).toByte() } }
                }

                override fun precomputeCondition(handle: Long, prompt: String, negative: String, width: Int, height: Int, clipSkip: Int): StableDiffusion.PrecomputedCondition? {
                    return StableDiffusion.PrecomputedCondition(
                        cCrossAttn = floatArrayOf(1.0f),
                        cCrossAttnDims = intArrayOf(1, 1),
                        cVector = floatArrayOf(1.0f),
                        cVectorDims = intArrayOf(1, 1),
                        cConcat = floatArrayOf(1.0f),
                        cConcatDims = intArrayOf(1, 1)
                    )
                }

                override fun txt2vidWithPrecomputedCondition(handle: Long, prompt: String, negative: String, width: Int, height: Int, videoFrames: Int, steps: Int, cfg: Float, seed: Long, sampleMethod: StableDiffusion.SampleMethod, scheduler: StableDiffusion.Scheduler, strength: Float, initImage: ByteArray?, initWidth: Int, initHeight: Int, cond: StableDiffusion.PrecomputedCondition?, uncond: StableDiffusion.PrecomputedCondition?, vaceStrength: Float, easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float, easyCacheStartPercent: Float, easyCacheEndPercent: Float): Array<ByteArray>? {
                    return Array(videoFrames) { ByteArray(width * height * 3) { 5 } }
                }

                override fun setProgressCallback(handle: Long, callback: StableDiffusion.VideoProgressCallback?) {}
                override fun cancelGeneration(handle: Long) {}
                override fun txt2ImgWithPrecomputedCondition(handle: Long, prompt: String, negative: String, width: Int, height: Int, steps: Int, cfg: Float, seed: Long, cond: StableDiffusion.PrecomputedCondition?, uncond: StableDiffusion.PrecomputedCondition?, easyCacheEnabled: Boolean, easyCacheReuseThreshold: Float, easyCacheStartPercent: Float, easyCacheEndPercent: Float): ByteArray? { return ByteArray(width * height * 3) { 3 } }
            }
        }

        // Intercept StableDiffusion.load to count calls and verify paths. Return fresh instances for each call
        var observedLoads = mutableListOf<Triple<String?, String?, String?>>()
        coEvery {
            StableDiffusion.load(any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any())
        } coAnswers {
            val callArgs = it.invocation.args
            // Kotlin 'load' signature: (context, modelId, filename, modelPath, vaePath, t5xxlPath, ...)
            val modelPathArg = callArgs[3] as String?
            val vaePathArg = callArgs[4] as String?
            val t5xxlArg = callArgs[5] as String?
            observedLoads.add(Triple(modelPathArg, vaePathArg, t5xxlArg))
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            val instance = constructor.newInstance(1L)
            // Update the metadata to indicate this is a video model
            instance.updateModelMetadata(
                StableDiffusion.VideoModelMetadata(
                    architecture = "Wan 2.1 T2V",
                    modelType = null,
                    parameterCount = "1.3B",
                    mobileSupported = true,
                    tags = setOf("wan-model"),
                    filename = modelFile.name,
                )
            )
            instance
        }

        // Run the flow
        val params = LLMEdgeManager.VideoGenerationParams(
            prompt = "test prompt",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 20,
            cfgScale = 7.0f,
            seed = 123L,
            flashAttn = true,
            forceSequentialLoad = true
        )

        val frames = LLMEdgeManager.generateVideo(context, params)
        assertNotNull("Should return frames", frames)
        assertTrue("Observed StableDiffusion.load should be called at least twice", observedLoads.size >= 2)

        // The final loaded diffusion model should not include a T5 path when loadT5=false
        assertNull("Loaded video model should not include T5 path", LLMEdgeManager.getLoadedVideoModelT5Path())

        // Verify at least one of the loads was for the T5 file and one of them for the model file
        assertTrue(observedLoads.any { it.first == t5File.absolutePath })
        assertTrue(observedLoads.any { it.first == modelFile.absolutePath })

        // The diffusion model load should have t5 arg == null because loadT5=false
        assertTrue(observedLoads.any { it.first == modelFile.absolutePath && it.third == null })

        // Verify generation returns expected number of frames
        assertEquals(params.videoFrames, frames.size)
        frames.forEach {
            assertEquals(params.width, it.width)
            assertEquals(params.height, it.height)
            assertEquals(Bitmap.Config.ARGB_8888, it.config)
        }

        // Reset bridge override
        StableDiffusion.resetNativeBridgeForTests()
    }
}
