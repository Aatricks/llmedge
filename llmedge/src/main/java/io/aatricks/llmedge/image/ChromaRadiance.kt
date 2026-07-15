package io.aatricks.llmedge.image

import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec

/**
 * Chroma Radiance presets for on-device generation.
 */
object ChromaRadiance {
    /** The Chroma Radiance DiT model. */
    @JvmField
    val diffusionModel: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "silveroxides/Chroma1-Radiance-GGUF",
            filename = "Chroma1-Radiance-v0.3/Chroma1-Radiance-v0.3-Q4_K_S.gguf",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /** The smaller Chroma1-HD Q3 DiT model intended for mobile devices. */
    @JvmField
    val mobileDiffusionModel: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "silveroxides/Chroma1-HD-GGUF",
            filename = "Chroma1-HD-Q3_K_S.gguf",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /** Quantized T5XXL text encoder shared by the mobile and Radiance presets. */
    @JvmField
    val t5xxl: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "city96/t5-v1_1-xxl-encoder-gguf",
            filename = "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.TEXT_ENCODER,
                    capabilities = setOf(ModelCapability.TEXT, ModelCapability.IMAGE),
                ),
        )

    /** The FLUX VAE model. */
    @JvmField
    val fluxVae: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "lodestones/Chroma",
            filename = "ae.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.VAE,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /**
     * Builds an [ImageGenerationRequest] pre-wired for Chroma Radiance.
     */
    @JvmStatic
    @JvmOverloads
    fun imageRequest(
        prompt: String,
        negative: String = "",
        width: Int = 512,
        height: Int = 512,
        steps: Int = 20,
        cfgScale: Float = 4.0f,
        seed: Long = -1L,
        flashAttention: Boolean = true,
        sequential: Boolean? = true,
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
            model = diffusionModel,
            t5xxl = t5xxl,
            splitDiffusionModel = true,
            sequential = sequential,
        )

    /**
     * Builds an [ImageGenerationRequest] pre-wired for the smaller Chroma1-HD Q3 model.
     */
    @JvmStatic
    @JvmOverloads
    fun mobileImageRequest(
        prompt: String,
        negative: String = "",
        width: Int = 512,
        height: Int = 512,
        steps: Int = 20,
        cfgScale: Float = 4.0f,
        seed: Long = -1L,
        flashAttention: Boolean = true,
        sequential: Boolean? = true,
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
            model = mobileDiffusionModel,
            vae = fluxVae,
            t5xxl = t5xxl,
            splitDiffusionModel = true,
            sequential = sequential,
        )
}
