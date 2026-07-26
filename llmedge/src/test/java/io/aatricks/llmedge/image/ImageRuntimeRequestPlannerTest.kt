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
        val params = ImageGenerationRequest(prompt = "x", textEncoder = cpuOnlyConfig.models.image, splitDiffusionModel = true)
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

    @Test
    fun `image request threads t5xxl slot from request to spec`() {
        val t5xxlSpec = io.aatricks.llmedge.model.ModelSpec.localFile(java.io.File("t5xxl.safetensors"))
        val request = ImageRuntimeRequestPlanner.imageRequest(
            ImageGenerationRequest(prompt = "x", t5xxl = t5xxlSpec),
            LLMEdgeConfig()
        )
        org.junit.Assert.assertEquals(t5xxlSpec, request.spec.t5xxl)
    }

    @Test
    fun `SD3 sequential planner has three phases with exact inclusion and exclusion, phase 1 CPU, phase 2 Vulkan-eligible`() {
        val t5xxlSpec = ModelSpec.LocalFile(File("t5xxl.safetensors"))
        val clipLSpec = ModelSpec.LocalFile(File("clip_l.safetensors"))
        val clipGSpec = ModelSpec.LocalFile(File("clip_g.safetensors"))
        val vaeSpec = ModelSpec.LocalFile(File("vae.safetensors"))
        val ditModelSpec = ModelSpec.LocalFile(File("dit.safetensors"))

        val params = ImageGenerationRequest(
            prompt = "x",
            model = ditModelSpec,
            vae = vaeSpec,
            t5xxl = t5xxlSpec,
            clipL = clipLSpec,
            clipG = clipGSpec,
            splitDiffusionModel = true,
            sequential = true
        )
        val plan = ImageRuntimeRequestPlanner.imageSequentialPlan(params, LLMEdgeConfig())

        // Phase 1: CLIP-only (CLIP-L and CLIP-G present, T5 absent, CPU)
        assertTrue(plan.conditioningRequest.spec.encoderOnly)
        assertFalse(plan.conditioningRequest.options.allowGpu)
        org.junit.Assert.assertEquals(clipLSpec, plan.conditioningRequest.spec.clipL)
        org.junit.Assert.assertEquals(clipGSpec, plan.conditioningRequest.spec.clipG)
        org.junit.Assert.assertNull(plan.conditioningRequest.spec.t5xxl)
        org.junit.Assert.assertEquals(clipLSpec, plan.conditioningRequest.spec.model)
        org.junit.Assert.assertNull(plan.conditioningRequest.spec.vae)

        // Phase 2: T5-only (T5 present, CLIPs absent, GPU-eligible when Vulkan enabled)
        val conditioningRequest2 = plan.conditioningRequest2
        org.junit.Assert.assertNotNull(conditioningRequest2)
        assertTrue(conditioningRequest2!!.spec.encoderOnly)
        assertTrue(conditioningRequest2.options.allowGpu)
        org.junit.Assert.assertNull(conditioningRequest2.spec.clipL)
        org.junit.Assert.assertNull(conditioningRequest2.spec.clipG)
        org.junit.Assert.assertEquals(t5xxlSpec, conditioningRequest2.spec.t5xxl)
        org.junit.Assert.assertEquals(t5xxlSpec, conditioningRequest2.spec.model)
        org.junit.Assert.assertNull(conditioningRequest2.spec.vae)

        // Phase 2 T5 remains CPU when useVulkan=false
        val cpuPlan = ImageRuntimeRequestPlanner.imageSequentialPlan(params, cpuOnlyConfig)
        val cpuConditioningRequest2 = cpuPlan.conditioningRequest2
        org.junit.Assert.assertNotNull(cpuConditioningRequest2)
        assertFalse(cpuConditioningRequest2!!.options.allowGpu)

        // Phase 3: DiT + VAE (DiT present, CLIP/T5 absent)
        org.junit.Assert.assertEquals(ditModelSpec, plan.diffusionRequest.spec.model)
        org.junit.Assert.assertEquals(vaeSpec, plan.diffusionRequest.spec.vae)
        org.junit.Assert.assertNull(plan.diffusionRequest.spec.clipL)
        org.junit.Assert.assertNull(plan.diffusionRequest.spec.clipG)
        org.junit.Assert.assertNull(plan.diffusionRequest.spec.t5xxl)
        assertTrue(plan.diffusionRequest.spec.splitDiffusionModel)
    }

    @Test
    fun `sequential image plan retains LoRA prompt resolution through every runtime`() {
        val params =
            ImageGenerationRequest(
                prompt = "x <lora:hyper:0.125>",
                model = ModelSpec.LocalFile(File("dit.safetensors")),
                vae = ModelSpec.LocalFile(File("vae.safetensors")),
                t5xxl = ModelSpec.LocalFile(File("t5xxl.safetensors")),
                clipL = ModelSpec.LocalFile(File("clip_l.safetensors")),
                clipG = ModelSpec.LocalFile(File("clip_g.safetensors")),
                loraModelDir = "/models/lora",
                loraApplyMode = io.aatricks.llmedge.image.diffusion.LoraApplyMode.IMMEDIATELY,
                splitDiffusionModel = true,
                sequential = true,
            )

        val plan = ImageRuntimeRequestPlanner.imageSequentialPlan(params, LLMEdgeConfig())

        org.junit.Assert.assertEquals("/models/lora", plan.conditioningRequest.options.loraModelDir)
        org.junit.Assert.assertEquals("/models/lora", plan.conditioningRequest2?.options?.loraModelDir)
        org.junit.Assert.assertEquals("/models/lora", plan.diffusionRequest.options.loraModelDir)
        org.junit.Assert.assertEquals(
            io.aatricks.llmedge.image.diffusion.LoraApplyMode.IMMEDIATELY,
            plan.diffusionRequest.options.loraApplyMode,
        )
    }

    @Test
    fun `masked T5 sequential planner isolates the text encoder before the diffusion model`() {
        val ditModel = ModelSpec.LocalFile(File("minit2i.safetensors"))
        val textEncoder = ModelSpec.LocalFile(File("flan-t5-large.safetensors"))
        val params =
            ImageGenerationRequest(
                prompt = "x",
                model = ditModel,
                textEncoder = textEncoder,
                diffusionModelOnly = true,
            )

        val plan =
            ImageRuntimeRequestPlanner.imageSequentialPlan(
                params = params,
                config = LLMEdgeConfig(),
                profile = ImageConditioningProfile.MASKED_T5,
            )

        assertTrue(plan.conditioningRequest.spec.encoderOnly)
        org.junit.Assert.assertEquals(textEncoder, plan.conditioningRequest.spec.model)
        org.junit.Assert.assertEquals(ImageConditioningProfile.MASKED_T5, plan.conditioningRequest.spec.conditioningProfile)
        assertTrue(plan.conditioningRequest.options.offloadToCpu)
        org.junit.Assert.assertEquals(ditModel, plan.diffusionRequest.spec.model)
        assertTrue(plan.diffusionRequest.spec.diffusionModelOnly)
        org.junit.Assert.assertNull(plan.diffusionRequest.spec.textEncoder)
        assertFalse(plan.diffusionRequest.options.offloadToCpu)
    }

    @Test
    fun `MiniT2I large request selects correct quant options in direct planning`() {
        val config = LLMEdgeConfig()
        val request = MiniT2I.largeImageRequest("test prompt")
        val plan = ImageRuntimeRequestPlanner.imageRequest(request, config)
        org.junit.Assert.assertEquals("q8_0", plan.options.weightType)
        org.junit.Assert.assertEquals(".*mask_token.*=f16", plan.options.tensorTypeRules)
    }

    @Test
    fun `MiniT2I large request selects correct quant options in sequential diffusion planning`() {
        val config = LLMEdgeConfig()
        val request = MiniT2I.largeImageRequest("test prompt")
        val plan = ImageRuntimeRequestPlanner.imageSequentialPlan(request, config)
        org.junit.Assert.assertEquals("q8_0", plan.diffusionRequest.options.weightType)
        org.junit.Assert.assertEquals(".*mask_token.*=f16", plan.diffusionRequest.options.tensorTypeRules)
    }

    @Test
    fun `MiniT2I large sequential conditioning options remain null`() {
        val config = LLMEdgeConfig()
        val request = MiniT2I.largeImageRequest("test prompt")
        val plan = ImageRuntimeRequestPlanner.imageSequentialPlan(request, config)
        org.junit.Assert.assertNull(plan.conditioningRequest.options.weightType)
        org.junit.Assert.assertNull(plan.conditioningRequest.options.tensorTypeRules)
    }

    @Test
    fun `MiniT2I equivalent HF spec selects correct quant options`() {
        val config = LLMEdgeConfig()
        val spec = ModelSpec.huggingFace(
            repoId = "MiniT2I/MiniT2I",
            filename = "minit2i-l-16/transformer/diffusion_pytorch_model.safetensors"
        )
        val request = MiniT2I.largeImageRequest("test prompt").copy(model = spec)
        val plan = ImageRuntimeRequestPlanner.imageRequest(request, config)
        org.junit.Assert.assertEquals("q8_0", plan.options.weightType)
        org.junit.Assert.assertEquals(".*mask_token.*=f16", plan.options.tensorTypeRules)
    }

    @Test
    fun `MiniT2I small or default request options remain null`() {
        val config = LLMEdgeConfig()
        val request = MiniT2I.imageRequest("test prompt")
        val plan = ImageRuntimeRequestPlanner.imageRequest(request, config)
        org.junit.Assert.assertNull(plan.options.weightType)
        org.junit.Assert.assertNull(plan.options.tensorTypeRules)

        val sequentialPlan = ImageRuntimeRequestPlanner.imageSequentialPlan(request, config)
        org.junit.Assert.assertNull(sequentialPlan.diffusionRequest.options.weightType)
        org.junit.Assert.assertNull(sequentialPlan.diffusionRequest.options.tensorTypeRules)
    }

    @Test
    fun `Chroma Radiance sequential planner has two phases with exact inclusion and exclusion`() {
        val t5xxlSpec = ModelSpec.LocalFile(File("t5xxl.safetensors"))
        val ditModelSpec = ModelSpec.LocalFile(File("dit.safetensors"))

        val params = ImageGenerationRequest(
            prompt = "x",
            model = ditModelSpec,
            t5xxl = t5xxlSpec,
            splitDiffusionModel = true,
            sequential = true
        )
        val plan = ImageRuntimeRequestPlanner.imageSequentialPlan(params, LLMEdgeConfig())

        // Phase 1: T5-only (T5 present, CLIPs absent, GPU-eligible when Vulkan enabled)
        assertTrue(plan.conditioningRequest.spec.encoderOnly)
        assertTrue(plan.conditioningRequest.options.allowGpu)
        org.junit.Assert.assertNull(plan.conditioningRequest.spec.clipL)
        org.junit.Assert.assertNull(plan.conditioningRequest.spec.clipG)
        org.junit.Assert.assertEquals(t5xxlSpec, plan.conditioningRequest.spec.t5xxl)
        org.junit.Assert.assertEquals(t5xxlSpec, plan.conditioningRequest.spec.model)
        org.junit.Assert.assertNull(plan.conditioningRequest.spec.vae)
        org.junit.Assert.assertNull(plan.conditioningRequest2)

        // Phase 2: DiT (DiT present, CLIP/T5/VAE absent)
        org.junit.Assert.assertEquals(ditModelSpec, plan.diffusionRequest.spec.model)
        org.junit.Assert.assertNull(plan.diffusionRequest.spec.vae)
        org.junit.Assert.assertNull(plan.diffusionRequest.spec.clipL)
        org.junit.Assert.assertNull(plan.diffusionRequest.spec.clipG)
        org.junit.Assert.assertNull(plan.diffusionRequest.spec.t5xxl)
        assertTrue(plan.diffusionRequest.spec.splitDiffusionModel)
    }
}
