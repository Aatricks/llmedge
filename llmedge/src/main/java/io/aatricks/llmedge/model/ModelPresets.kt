package io.aatricks.llmedge.model

/**
 * Ready-to-use [ModelSpec] presets for models that run well on low-end devices and are supported by the
 * bundled ik_llama.cpp runtime.
 *
 * Pass them directly to inference calls, or wire them into [ModelRegistry] / `LLMEdgeConfig`:
 *
 * ```kotlin
 * val reply = edge.text.generate(prompt = "Hi", model = ModelPresets.bitnet)
 *
 * val caption = edge.vision.analyze(
 *     image = bitmap,
 *     prompt = "Describe this image.",
 *     model = ModelPresets.smolVlm2.model,
 *     projector = ModelPresets.smolVlm2.projector,
 * )
 * ```
 */
object ModelPresets {
    /**
     * Microsoft **BitNet b1.58 2B4T** — native 1-bit LLM (`IQ2_BN_R4`, ~988 MB) for the ik_llama.cpp
     * runtime. The correct chat template ships on the spec ([ModelHints.chatTemplate]), so generation
     * is well-formed without manually setting `TextModelOptions.chatTemplate`.
     */
    val bitnet: ModelSpec = DefaultModelCatalog.bitnetText

    /**
     * **SmolVLM2-256M** vision model — base + projector (~280 MB total). A tiny multimodal model for
     * on-device image understanding through the mtmd/clip path.
     */
    val smolVlm2: VisionModels =
        VisionModels(
            model = DefaultModelCatalog.smolVlm2Model,
            projector = DefaultModelCatalog.smolVlm2Projector,
        )
}
