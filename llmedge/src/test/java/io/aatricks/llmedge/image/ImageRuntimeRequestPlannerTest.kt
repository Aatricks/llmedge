package io.aatricks.llmedge.image

import io.aatricks.llmedge.ImageRuntimeConfig
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.model.ModelSpec
import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ImageRuntimeRequestPlannerTest {
    private val cpuOnlyConfig = LLMEdgeConfig(image = ImageRuntimeConfig(useVulkan = false))

    @Test
    fun `image request keeps gpu eligible by default`() {
        val request = ImageRuntimeRequestPlanner.imageRequest(ImageGenerationRequest(prompt = "x"), LLMEdgeConfig())
        assertTrue(request.options.allowGpu)
    }

    @Test
    fun `image request forces cpu when config disables vulkan`() {
        val request = ImageRuntimeRequestPlanner.imageRequest(ImageGenerationRequest(prompt = "x"), cpuOnlyConfig)
        assertFalse(request.options.allowGpu)
    }

    @Test
    fun `sequential image plan forces cpu when config disables vulkan`() {
        val params = ImageGenerationRequest(prompt = "x", textEncoder = cpuOnlyConfig.models.image)
        val plan = ImageRuntimeRequestPlanner.imageSequentialPlan(params, cpuOnlyConfig)
        assertFalse(plan.diffusionRequest.options.allowGpu)
        // The conditioning phase is CPU-pinned regardless of config (peak-RAM promise).
        assertFalse(plan.conditioningRequest.options.allowGpu)
    }

    @Test
    fun `standalone diffusion model route is retained in runtime spec`() {
        val model = ModelSpec.LocalFile(File("/minit2i.safetensors"))
        val request =
            ImageRuntimeRequestPlanner.imageRequest(
                ImageGenerationRequest(
                    prompt = "x",
                    model = model,
                    diffusionModelOnly = true,
                ),
                LLMEdgeConfig(),
            )

        assertSame(model, request.spec.model)
        assertTrue(request.spec.diffusionModelOnly)
        assertFalse(request.spec.splitDiffusionModel)
    }

    @Test
    fun `video request forces cpu when config disables vulkan`() {
        val defaultRequest = ImageRuntimeRequestPlanner.directVideoRequest(VideoGenerationRequest(prompt = "x"), LLMEdgeConfig())
        assertTrue(defaultRequest.options.allowGpu)

        val cpuRequest = ImageRuntimeRequestPlanner.directVideoRequest(VideoGenerationRequest(prompt = "x"), cpuOnlyConfig)
        assertFalse(cpuRequest.options.allowGpu)
    }

    @Test
    fun `image request threads clip slots from request to spec`() {
        val clipLSpec = io.aatricks.llmedge.model.ModelSpec.localFile(java.io.File("clip_l.safetensors"))
        val clipGSpec = io.aatricks.llmedge.model.ModelSpec.localFile(java.io.File("clip_g.safetensors"))
        val request = ImageRuntimeRequestPlanner.imageRequest(
            ImageGenerationRequest(prompt = "x", clipL = clipLSpec, clipG = clipGSpec),
            LLMEdgeConfig()
        )
        org.junit.Assert.assertEquals(clipLSpec, request.spec.clipL)
        org.junit.Assert.assertEquals(clipGSpec, request.spec.clipG)
        org.junit.Assert.assertNull(request.spec.vae)
    }

    @Test
    fun `video request threads highNoiseDiffusionModel from request to spec`() {
        val highNoiseSpec = io.aatricks.llmedge.model.ModelSpec.localFile(java.io.File("high_noise.safetensors"))
        val request = ImageRuntimeRequestPlanner.directVideoRequest(
            VideoGenerationRequest(prompt = "x", highNoiseDiffusionModel = highNoiseSpec),
            LLMEdgeConfig()
        )
        org.junit.Assert.assertEquals(highNoiseSpec, request.spec.highNoiseDiffusionModel)
    }
}
