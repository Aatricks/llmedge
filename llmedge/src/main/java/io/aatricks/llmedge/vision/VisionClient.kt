package io.aatricks.llmedge.vision

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.vision.ocr.MlKitOcrEngine

data class VisionRequest(
    val image: Bitmap,
    val prompt: String,
    val model: ModelSpec,
    val projector: ModelSpec,
    /** Prompt/batch processing threads for the underlying SmolLM runtime. */
    val numThreads: Int? = null,
    /** Single-token generation threads for the underlying SmolLM runtime. */
    val generationThreads: Int? = null,
)

class VisionClient internal constructor(
    private val context: Context,
    private val pipeline: VisionPipeline,
    config: LLMEdgeConfig,
) : AutoCloseable {
    private val defaultModel: ModelSpec = config.models.vision.model
    private val defaultProjector: ModelSpec = config.models.vision.projector
    private val defaultPromptThreads: Int = config.defaultTextThreads.coerceAtLeast(1)
    private val defaultGenerationThreads: Int =
        config.defaultTextGenerationThreads.coerceAtLeast(1)

    /**
     * Analyze an image with the configured vision pipeline.
     *
      * This VLM path is experimental. It requires both a vision-capable GGUF and a compatible
      * projector/mmproj file; otherwise the call fails fast with a descriptive error. For the more
      * stable OCR-only path, use [extractText].
      *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when the selected vision components cannot
     * be resolved or invoked.
     */
    suspend fun analyze(
        request: VisionRequest,
        onStatus: ((String) -> Unit)? = null,
    ): String = pipeline.analyze(request, onStatus)

    suspend fun analyze(
        image: Bitmap,
        prompt: String,
        model: ModelSpec = defaultModel,
        projector: ModelSpec = defaultProjector,
        numThreads: Int = defaultPromptThreads,
        generationThreads: Int = defaultGenerationThreads,
        onStatus: ((String) -> Unit)? = null,
    ): String =
        analyze(
            VisionRequest(
                image = image,
                prompt = prompt,
                model = model,
                projector = projector,
                numThreads = numThreads,
                generationThreads = generationThreads,
            ),
            onStatus,
        )

    /**
     * Extract text with the bundled OCR pipeline.
     *
     * This is independent from VLM-based analysis and works without loading a vision-language model.
     */
    suspend fun extractText(image: Bitmap): String {
        val engine = MlKitOcrEngine(context)
        try {
            return engine.extractText(ImageSource.BitmapSource(image)).text
        } finally {
            engine.close()
        }
    }

    override fun close() = Unit
}
