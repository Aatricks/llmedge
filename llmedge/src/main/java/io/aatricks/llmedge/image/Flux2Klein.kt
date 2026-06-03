package io.aatricks.llmedge.image

import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec

/**
 * FLUX.2 Klein 4B — a distilled diffusion-transformer image-generation model that runs on-device
 * through stable-diffusion.cpp. It is delivered as three separate files instead of one checkpoint:
 *
 *  1. the diffusion transformer (DiT) GGUF,
 *  2. a Qwen3-4B text encoder, and
 *  3. the FLUX.2 autoencoder (VAE).
 *
 * This is the same FLUX.2 Klein 4B architecture that PrismML's binary/ternary "Bonsai Image"
 * models are built on. Bonsai's own 1-bit/ternary releases ship only in MLX (Apple) and GemLite
 * (CUDA) packings, neither of which loads on Android; this GGUF build is the Android-runnable
 * equivalent at a comparable footprint.
 *
 * Footprint: ~2.5 GB DiT (Q4_0) + ~3.85 GB encoder (fp4) + ~0.32 GB VAE. This targets high-RAM
 * devices; the runtime offloads weights to CPU automatically for split models (see
 * [ImageGenerationRequest.splitDiffusionModel]).
 *
 * Usage:
 * ```
 * val bitmap = imageClient.generate(Flux2Klein.imageRequest("a red fox in snow, 8k"))
 * ```
 */
object Flux2Klein {
    /** The FLUX.2 Klein 4B diffusion transformer (routed to `diffusion_model_path`). */
    @JvmField
    val diffusionModel: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "leejet/FLUX.2-klein-4B-GGUF",
            filename = "flux-2-klein-4b-Q4_0.gguf",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /**
     * PrismML's **Bonsai Image** ternary DiT (FLUX.2 Klein 4B QAT), quantized to Q2_K GGUF
     * (~1.3 GB) — a smaller, quality-preserving alternative to [diffusionModel]. Pair with
     * [textEncoder] + [vae]; ideal with `sequential = true` for ~4 GB-RAM devices.
     */
    @JvmField
    val bonsaiDiffusionModel: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "Aatricks/bonsai-image-ternary-4B-FLUX2-klein-GGUF",
            filename = "bonsai-flux2-klein-ternary-q2_k.gguf",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /**
     * The Qwen3-4B text encoder (routed to `llm_path`), Q3_K_M GGUF (~2.1 GB). sdcpp's FLUX.2 LLM
     * loader reads this standard GGUF directly; Comfy's fp4 safetensors packing is NOT compatible
     * (sdcpp misreads its intermediate_size), so a GGUF-quantized encoder is required here.
     *
     * Sourced from a community GGUF repackaging of Qwen3-4B. If that repo becomes unavailable, the
     * canonical fallback is to GGUF-quantize `Comfy-Org/vae-text-encorder-for-flux-klein-4b`'s
     * `qwen_3_4b.safetensors` (fp16) with stable-diffusion.cpp's `-M convert --type q4_0`, then
     * point [textEncoder] at the result.
     */
    @JvmField
    val textEncoder: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            filename = "Qwen_3_4b-Q3_K_M.gguf",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.TEXT_ENCODER,
                    capabilities = setOf(ModelCapability.TEXT, ModelCapability.IMAGE),
                ),
        )

    /** The FLUX.2 autoencoder. */
    @JvmField
    val vae: ModelSpec =
        ModelSpec.huggingFace(
            repoId = "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            filename = "split_files/vae/flux2-vae.safetensors",
            preferredQuantizations = emptyList(),
            hints =
                ModelHints(
                    artifactKind = ModelArtifactKind.VAE,
                    capabilities = setOf(ModelCapability.IMAGE),
                ),
        )

    /**
     * Build an [ImageGenerationRequest] pre-wired for FLUX.2 Klein 4B: the three component models,
     * [ImageGenerationRequest.splitDiffusionModel] enabled, and the distilled-model generation
     * defaults (CFG 1.0, 4 steps). Override [steps]/[cfgScale] for quality/speed trade-offs.
     */
    @JvmStatic
    @JvmOverloads
    fun imageRequest(
        prompt: String,
        negative: String = "",
        width: Int = 512,
        height: Int = 512,
        steps: Int = 4,
        cfgScale: Float = 1.0f,
        seed: Long = -1L,
        flashAttention: Boolean = true,
        sequential: Boolean = false,
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
            textEncoder = textEncoder,
            splitDiffusionModel = true,
            sequential = sequential,
        )

    /**
     * Low-memory request: PrismML's Bonsai ternary DiT ([bonsaiDiffusionModel], ~1.3 GB) with
     * [sequential] loading on by default, so peak RAM ≈ max(encoder, DiT) ≈ 2.6 GB — targeting
     * ~4 GB-RAM devices.
     */
    @JvmStatic
    @JvmOverloads
    fun bonsaiImageRequest(
        prompt: String,
        negative: String = "",
        width: Int = 512,
        height: Int = 512,
        steps: Int = 4,
        cfgScale: Float = 1.0f,
        seed: Long = -1L,
        flashAttention: Boolean = true,
        sequential: Boolean = true,
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
            model = bonsaiDiffusionModel,
            vae = vae,
            textEncoder = textEncoder,
            splitDiffusionModel = true,
            sequential = sequential,
        )
}
