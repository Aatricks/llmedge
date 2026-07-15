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

    /** T5XXL FP8 text encoder. */
    @JvmField
    val t5xxl: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "comfyanonymous/flux_text_encoders",
            filename = "t5xxl_fp8_e4m3fn.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.TEXT_ENCODER,
                    capabilities = setOf(ModelCapability.TEXT, ModelCapability.IMAGE),
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
}
