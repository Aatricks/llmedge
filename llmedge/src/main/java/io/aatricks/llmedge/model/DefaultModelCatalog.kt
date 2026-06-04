package io.aatricks.llmedge.model

import io.aatricks.llmedge.huggingface.HuggingFaceHub

internal object DefaultModelCatalog {
    val text: ModelSpec =
        huggingFaceSpec(
            // HuggingFaceTB/SmolLM-135M-Instruct-GGUF is unavailable to anonymous clients (HTTP 401),
            // which made every default-text consumer (e.g. the RAG session) fail to build. Use the
            // public MaziyarPanahi mirror, which serves the same model.
            repoId = "MaziyarPanahi/SmolLM-135M-Instruct-GGUF",
            filename = "SmolLM-135M-Instruct.Q4_K_M.gguf",
            artifactKind = ModelArtifactKind.GGUF_MODEL,
            capabilities = setOf(ModelCapability.TEXT),
        )

    val speechToText: ModelSpec =
        huggingFaceSpec(
            repoId = "ggerganov/whisper.cpp",
            filename = "ggml-tiny.bin",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.REPO_FILE,
            capabilities = setOf(ModelCapability.SPEECH_TO_TEXT),
        )

    val textToSpeech: ModelSpec =
        huggingFaceSpec(
            repoId = "Green-Sky/bark-ggml",
            filename = "bark-small_weights-f16.bin",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.REPO_FILE,
            capabilities = setOf(ModelCapability.TEXT_TO_SPEECH),
        )

    val image: ModelSpec =
        huggingFaceSpec(
            repoId = "Meina/MeinaMix",
            filename = "MeinaPastel - baked VAE.safetensors",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
            capabilities = setOf(ModelCapability.IMAGE),
        )

    val visionModel: ModelSpec =
        huggingFaceSpec(
            repoId = "xtuner/llava-phi-3-mini-gguf",
            filename = "llava-phi-3-mini-int4.gguf",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.GGUF_MODEL,
            capabilities = setOf(ModelCapability.TEXT, ModelCapability.VISION),
        )

    val visionProjector: ModelSpec =
        huggingFaceSpec(
            repoId = "xtuner/llava-phi-3-mini-gguf",
            filename = "llava-phi-3-mini-mmproj-f16.gguf",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.PROJECTOR,
            capabilities = setOf(ModelCapability.PROJECTOR),
        )

    /**
     * Microsoft BitNet b1.58 2B4T — native 1-bit LLM (IQ2_BN for ik_llama.cpp). ~988 MB.
     * Plain IQ2_BN is the portable default; the same repo also ships an `_r4` CPU row-interleaved
     * repack (`bitnet1582b4t-iq2_bn_r4.gguf`) that can be faster on pure-CPU setups.
     */
    val bitnetText: ModelSpec =
        huggingFaceSpec(
            repoId = "tdh111/bitnet-b1.58-2B-4T-GGUF",
            filename = "bitnet1582b4t-iq2_bn.gguf",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.GGUF_MODEL,
            capabilities = setOf(ModelCapability.TEXT),
            chatTemplate = ModelChatTemplates.BITNET,
        )

    /** SmolVLM2-256M vision base model (loaded together with [smolVlm2Projector]). ~175 MB. */
    val smolVlm2Model: ModelSpec =
        huggingFaceSpec(
            repoId = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
            filename = "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.GGUF_MODEL,
            capabilities = setOf(ModelCapability.TEXT, ModelCapability.VISION),
        )

    /** SmolVLM2-256M multimodal projector (mmproj) paired with [smolVlm2Model]. ~104 MB. */
    val smolVlm2Projector: ModelSpec =
        huggingFaceSpec(
            repoId = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
            filename = "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.PROJECTOR,
            capabilities = setOf(ModelCapability.PROJECTOR),
        )

    val videoDiffusion: ModelSpec =
        huggingFaceSpec(
            repoId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            filename = "wan2.1_t2v_1.3B_fp16.safetensors",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
            capabilities = setOf(ModelCapability.IMAGE, ModelCapability.VIDEO),
        )

    val videoVae: ModelSpec =
        huggingFaceSpec(
            repoId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            filename = "wan_2.1_vae.safetensors",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.VAE,
            capabilities = setOf(ModelCapability.VIDEO),
        )

    val videoTextEncoder: ModelSpec =
        huggingFaceSpec(
            repoId = "city96/umt5-xxl-encoder-gguf",
            filename = "umt5-xxl-encoder-Q3_K_S.gguf",
            preferredQuantizations = emptyList(),
            artifactKind = ModelArtifactKind.TEXT_ENCODER,
            capabilities = setOf(ModelCapability.TEXT, ModelCapability.VIDEO),
        )

    private fun huggingFaceSpec(
        repoId: String,
        filename: String,
        preferredQuantizations: List<String> = HuggingFaceHub.DEFAULT_QUANTIZATION_PRIORITIES,
        artifactKind: ModelArtifactKind,
        capabilities: Set<ModelCapability>,
        chatTemplate: String? = null,
    ): ModelSpec =
        ModelSpec.huggingFace(
            repoId = repoId,
            filename = filename,
            preferredQuantizations = preferredQuantizations,
            hints =
                ModelHints(
                    artifactKind = artifactKind,
                    capabilities = capabilities,
                    chatTemplate = chatTemplate,
                ),
        )
}
