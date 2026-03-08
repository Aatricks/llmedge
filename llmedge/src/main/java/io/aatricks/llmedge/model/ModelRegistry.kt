package io.aatricks.llmedge.model

data class VisionModels(
    val model: ModelSpec = ModelSpec.huggingFace(
        repoId = "xtuner/llava-phi-3-mini-gguf",
        filename = "llava-phi-3-mini-int4.gguf",
        preferredQuantizations = emptyList(),
    ),
    val projector: ModelSpec = ModelSpec.huggingFace(
        repoId = "xtuner/llava-phi-3-mini-gguf",
        filename = "llava-phi-3-mini-mmproj-f16.gguf",
        preferredQuantizations = emptyList(),
    ),
)

data class VideoModels(
    val diffusion: ModelSpec = ModelSpec.huggingFace(
        repoId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename = "wan2.1_t2v_1.3B_fp16.safetensors",
        preferredQuantizations = emptyList(),
    ),
    val vae: ModelSpec = ModelSpec.huggingFace(
        repoId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename = "wan_2.1_vae.safetensors",
        preferredQuantizations = emptyList(),
    ),
    val textEncoder: ModelSpec = ModelSpec.huggingFace(
        repoId = "city96/umt5-xxl-encoder-gguf",
        filename = "umt5-xxl-encoder-Q3_K_S.gguf",
        preferredQuantizations = emptyList(),
    ),
)

data class ModelRegistry(
    val text: ModelSpec = ModelSpec.huggingFace(
        repoId = "HuggingFaceTB/SmolLM-135M-Instruct-GGUF",
        filename = "smollm-135m-instruct.q4_k_m.gguf",
    ),
    val speechToText: ModelSpec = ModelSpec.huggingFace(
        repoId = "ggerganov/whisper.cpp",
        filename = "ggml-tiny.bin",
        preferredQuantizations = emptyList(),
    ),
    val textToSpeech: ModelSpec = ModelSpec.huggingFace(
        repoId = "Green-Sky/bark-ggml",
        filename = "bark-small_weights-f16.bin",
        preferredQuantizations = emptyList(),
    ),
    val image: ModelSpec = ModelSpec.huggingFace(
        repoId = "Meina/MeinaMix",
        filename = "MeinaPastel - baked VAE.safetensors",
        preferredQuantizations = emptyList(),
    ),
    val vision: VisionModels = VisionModels(),
    val video: VideoModels = VideoModels(),
)
