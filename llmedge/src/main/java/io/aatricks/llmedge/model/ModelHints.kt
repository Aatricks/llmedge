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
    /**
     * Optional chat template (Jinja source) the model should use when the caller does not supply one
     * via inference options. Lets a preset stay self-contained for models whose GGUF metadata carries a
     * missing or incorrect template (e.g. BitNet b1.58). A caller-provided template always wins.
     */
    val chatTemplate: String? = null,
) {
    fun hasCapability(capability: ModelCapability): Boolean = capability in capabilities
}
