package io.aatricks.llmedge.model

enum class ModelArtifactKind {
    AUTO,
    GGUF_MODEL,
    REPO_FILE,
    PROJECTOR,
    DIFFUSION_MODEL,
    VAE,
    TEXT_ENCODER,
    TAEHV,
}

enum class ModelCapability {
    TEXT,
    VISION,
    PROJECTOR,
    IMAGE,
    VIDEO,
    SPEECH_TO_TEXT,
    TEXT_TO_SPEECH,
}

data class ModelHints(
    val artifactKind: ModelArtifactKind = ModelArtifactKind.AUTO,
    val capabilities: Set<ModelCapability> = emptySet(),
) {
    fun hasCapability(capability: ModelCapability): Boolean = capability in capabilities
}
