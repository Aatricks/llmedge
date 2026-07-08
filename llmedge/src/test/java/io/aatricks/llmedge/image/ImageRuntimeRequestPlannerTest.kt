package io.aatricks.llmedge.image

import io.aatricks.llmedge.ImageRuntimeConfig
import io.aatricks.llmedge.LLMEdgeConfig
import org.junit.Assert.assertFalse
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
    fun `video request forces cpu when config disables vulkan`() {
        val defaultRequest = ImageRuntimeRequestPlanner.directVideoRequest(VideoGenerationRequest(prompt = "x"), LLMEdgeConfig())
        assertTrue(defaultRequest.options.allowGpu)

        val cpuRequest = ImageRuntimeRequestPlanner.directVideoRequest(VideoGenerationRequest(prompt = "x"), cpuOnlyConfig)
        assertFalse(cpuRequest.options.allowGpu)
    }
}
