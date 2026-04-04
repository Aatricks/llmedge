package io.aatricks.llmedge.model

data class VisionModels(
    val model: ModelSpec = DefaultModelCatalog.visionModel,
    val projector: ModelSpec = DefaultModelCatalog.visionProjector,
)

data class VideoModels(
    val diffusion: ModelSpec = DefaultModelCatalog.videoDiffusion,
    val vae: ModelSpec = DefaultModelCatalog.videoVae,
    val textEncoder: ModelSpec = DefaultModelCatalog.videoTextEncoder,
)

data class ModelRegistry(
    val text: ModelSpec = DefaultModelCatalog.text,
    val speechToText: ModelSpec = DefaultModelCatalog.speechToText,
    val textToSpeech: ModelSpec = DefaultModelCatalog.textToSpeech,
    val image: ModelSpec = DefaultModelCatalog.image,
    val vision: VisionModels = VisionModels(),
    val video: VideoModels = VideoModels(),
)
