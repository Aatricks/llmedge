package io.aatricks.llmedge.image

import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec

/**
 * Stable Diffusion 3 Medium presets for on-device generation through stable-diffusion.cpp.
 *
 * This preset skips the optional T5xxl text encoder (stable-diffusion.cpp probes clip_l,
 * clip_g, and t5 independently) — a quality tradeoff that avoids the ~5 GB T5 download.
 * The sequential low-RAM mode is unsupported for SD3 because the encoder-only native load/unload
 * path is Flux2/T5-specific in stable-diffusion.cpp.
 *
 * Mobile default resolution is 512x512, though the model's native resolution is 1024x1024.
 * Callers with sufficient RAM headroom may pass 1024x1024.
 *
 * The total split download size is ~3.1 GB.
 */
object Sd3Medium {
    /** The SD3 Medium DiT model (Q4_0, ~1.28 GB). */
    @JvmField
    val diffusionModel: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "city96/stable-diffusion-3-medium-gguf",
            filename = "sd3_medium-Q4_0.gguf",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /** CLIP-L text encoder (fp16/fp8 source, ~250 MB). */
    @JvmField
    val clipL: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "Comfy-Org/stable-diffusion-3.5-fp8",
            filename = "text_encoders/clip_l.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.TEXT_ENCODER,
                    capabilities = setOf(ModelCapability.TEXT, ModelCapability.IMAGE),
                ),
        )

    /** CLIP-G text encoder (fp16/fp8 source, ~1.39 GB). */
    @JvmField
    val clipG: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "Comfy-Org/stable-diffusion-3.5-fp8",
            filename = "text_encoders/clip_g.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.TEXT_ENCODER,
                    capabilities = setOf(ModelCapability.TEXT, ModelCapability.IMAGE),
                ),
        )

    /** SD3/SD3.5 16-channel VAE (~168 MB). */
    @JvmField
    val vae: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "Shio-Koube/SD-3.5-vae",
            filename = "diffusion_pytorch_model.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.VAE,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /** SD3 Medium All-in-One Model (encoders and VAE baked in, Q4_0, ~4.55 GB). */
    @JvmField
    val allInOneModel: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "second-state/stable-diffusion-3-medium-GGUF",
            filename = "sd3-medium-Q4_0.gguf",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /**
     * Builds an [ImageGenerationRequest] for Stable Diffusion 3 Medium using split models
     * (diffusionModel, clipL, clipG, and vae).
     */
    @JvmStatic
    @JvmOverloads
    fun imageRequest(
        prompt: String,
        negative: String = "",
        width: Int = 512,
        height: Int = 512,
        steps: Int = 28,
        cfgScale: Float = 4.5f,
        seed: Long = -1L,
        flashAttention: Boolean = true,
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
            vae = vae,
            clipL = clipL,
            clipG = clipG,
            textEncoder = null,
            splitDiffusionModel = true,
            sequential = false,
        )

    /**
     * Builds an [ImageGenerationRequest] for Stable Diffusion 3 Medium using the all-in-one model.
     */
    @JvmStatic
    @JvmOverloads
    fun allInOneImageRequest(
        prompt: String,
        negative: String = "",
        width: Int = 512,
        height: Int = 512,
        steps: Int = 28,
        cfgScale: Float = 4.5f,
        seed: Long = -1L,
        flashAttention: Boolean = true,
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
            model = allInOneModel,
            vae = null,
            clipL = null,
            clipG = null,
            textEncoder = null,
            splitDiffusionModel = false,
            sequential = false,
        )
}
