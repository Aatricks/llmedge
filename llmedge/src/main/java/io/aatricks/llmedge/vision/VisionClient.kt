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
)

class VisionClient internal constructor(
    private val context: Context,
    private val pipeline: VisionPipeline,
    config: LLMEdgeConfig,
) : AutoCloseable {
    private val defaultModel: ModelSpec = config.models.vision.model
    private val defaultProjector: ModelSpec = config.models.vision.projector

    suspend fun analyze(
        request: VisionRequest,
        onStatus: ((String) -> Unit)? = null,
    ): String = pipeline.analyze(request, onStatus)

    suspend fun analyze(
        image: Bitmap,
        prompt: String,
        model: ModelSpec = defaultModel,
        projector: ModelSpec = defaultProjector,
        onStatus: ((String) -> Unit)? = null,
    ): String = analyze(VisionRequest(image, prompt, model, projector), onStatus)

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
