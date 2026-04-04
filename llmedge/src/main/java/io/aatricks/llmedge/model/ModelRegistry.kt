package io.aatricks.llmedge.model

data class VisionModels(
    val model: ModelSpec = ModelSpec.huggingFace(
        repoId = "xtuner/llava-phi-3-mini-gguf",
        filename = "llava-phi-3-mini-int4.gguf",
        preferredQuantizations = emptyList(),
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.GGUF_MODEL,
                capabilities = setOf(ModelCapability.TEXT, ModelCapability.VISION),
            ),
    ),
    val projector: ModelSpec = ModelSpec.huggingFace(
        repoId = "xtuner/llava-phi-3-mini-gguf",
        filename = "llava-phi-3-mini-mmproj-f16.gguf",
        preferredQuantizations = emptyList(),
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.PROJECTOR,
                capabilities = setOf(ModelCapability.PROJECTOR),
            ),
    ),
)

data class VideoModels(
    val diffusion: ModelSpec = ModelSpec.huggingFace(
        repoId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename = "wan2.1_t2v_1.3B_fp16.safetensors",
        preferredQuantizations = emptyList(),
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                capabilities = setOf(ModelCapability.IMAGE, ModelCapability.VIDEO),
            ),
    ),
    val vae: ModelSpec = ModelSpec.huggingFace(
        repoId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename = "wan_2.1_vae.safetensors",
        preferredQuantizations = emptyList(),
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.VAE,
                capabilities = setOf(ModelCapability.VIDEO),
            ),
    ),
    val textEncoder: ModelSpec = ModelSpec.huggingFace(
        repoId = "city96/umt5-xxl-encoder-gguf",
        filename = "umt5-xxl-encoder-Q3_K_S.gguf",
        preferredQuantizations = emptyList(),
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.TEXT_ENCODER,
                capabilities = setOf(ModelCapability.TEXT, ModelCapability.VIDEO),
            ),
    ),
)

data class ModelRegistry(
    val text: ModelSpec = ModelSpec.huggingFace(
        repoId = "HuggingFaceTB/SmolLM-135M-Instruct-GGUF",
        filename = "smollm-135m-instruct.q4_k_m.gguf",
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.GGUF_MODEL,
                capabilities = setOf(ModelCapability.TEXT),
            ),
    ),
    val speechToText: ModelSpec = ModelSpec.huggingFace(
        repoId = "ggerganov/whisper.cpp",
        filename = "ggml-tiny.bin",
        preferredQuantizations = emptyList(),
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.REPO_FILE,
                capabilities = setOf(ModelCapability.SPEECH_TO_TEXT),
            ),
    ),
    val textToSpeech: ModelSpec = ModelSpec.huggingFace(
        repoId = "Green-Sky/bark-ggml",
        filename = "bark-small_weights-f16.bin",
        preferredQuantizations = emptyList(),
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.REPO_FILE,
                capabilities = setOf(ModelCapability.TEXT_TO_SPEECH),
            ),
    ),
    val image: ModelSpec = ModelSpec.huggingFace(
        repoId = "Meina/MeinaMix",
        filename = "MeinaPastel - baked VAE.safetensors",
        preferredQuantizations = emptyList(),
        hints =
            ModelHints(
                artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                capabilities = setOf(ModelCapability.IMAGE),
            ),
    ),
    val vision: VisionModels = VisionModels(),
    val video: VideoModels = VideoModels(),
)
