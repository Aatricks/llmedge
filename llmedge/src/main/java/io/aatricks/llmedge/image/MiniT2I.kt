package io.aatricks.llmedge.image

import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec

/** MiniT2I image generation with the FLAN-T5 Large text encoder. */
object MiniT2I {
    @JvmField
    val diffusionModel: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "MiniT2I/MiniT2I",
            filename = "minit2i-b-16/transformer/diffusion_pytorch_model.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    @JvmField
    val diffusionModelLarge: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "MiniT2I/MiniT2I",
            filename = "minit2i-l-16/transformer/diffusion_pytorch_model.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    @JvmField
    val textEncoder: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "google/flan-t5-large",
            filename = "model.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.TEXT_ENCODER,
                    capabilities = setOf(ModelCapability.TEXT, ModelCapability.IMAGE),
                ),
        )

    @JvmStatic
    @JvmOverloads
    fun imageRequest(
        prompt: String,
        negative: String = "",
        width: Int = 512,
        height: Int = 512,
        steps: Int = 100,
        cfgScale: Float = 6.0f,
        seed: Long = -1L,
        flashAttention: Boolean = true,
    ): ImageGenerationRequest =
        buildRequest(
            model = diffusionModel,
            prompt = prompt,
            negative = negative,
            width = width,
            height = height,
            steps = steps,
            cfgScale = cfgScale,
            seed = seed,
            flashAttention = flashAttention,
        )

    @JvmStatic
    @JvmOverloads
    fun largeImageRequest(
        prompt: String,
        negative: String = "",
        width: Int = 512,
        height: Int = 512,
        steps: Int = 100,
        cfgScale: Float = 6.0f,
        seed: Long = -1L,
        flashAttention: Boolean = true,
    ): ImageGenerationRequest =
        buildRequest(
            model = diffusionModelLarge,
            prompt = prompt,
            negative = negative,
            width = width,
            height = height,
            steps = steps,
            cfgScale = cfgScale,
            seed = seed,
            flashAttention = flashAttention,
        )

    private fun buildRequest(
        model: ModelSpec,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        steps: Int,
        cfgScale: Float,
        seed: Long,
        flashAttention: Boolean,
    ): ImageGenerationRequest =
        ImageGenerationRequest(
            prompt = prompt,
            negative = negative,
            width = width,
            height = height,
            steps = steps,
            cfgScale = cfgScale,
            seed = seed,
            flashAttention = flashAttention,
            model = model,
            textEncoder = textEncoder,
            diffusionModelOnly = true,
        )
}
