package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.SmolLM
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

internal class VisionPipeline(
    private val context: Context,
    private val resolver: ModelResolver,
) {
    suspend fun analyze(
        request: VisionRequest,
        onStatus: ((String) -> Unit)? = null,
    ): String =
        withContext(Dispatchers.IO) {
            val modelFile = resolver.resolve(context, request.model)
            val projectorFile = resolver.resolve(context, request.projector)
            val smol = SmolLM(useVulkan = false)

            try {
                onStatus?.invoke("Preparing image")
                val adapter = SmolLMVisionAdapter(context, smol)
                try {
                    val imageFile = File.createTempFile("vision_input", ".jpg", context.cacheDir)
                    val embedFile = File.createTempFile("vision_prepared", ".bin", context.cacheDir)
                    val metaFile = File(embedFile.absolutePath + ".meta.json")

                    try {
                        val scaled =
                            ImageUtils.preprocessBitmap(
                                request.image,
                                maxDimension = 672,
                                enhance = false,
                            )
                        imageFile.outputStream().use { out ->
                            scaled.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
                        }

                        onStatus?.invoke("Loading vision model")
                        smol.load(
                            modelPath = modelFile.absolutePath,
                            params =
                                SmolLM.InferenceParams(
                                    numThreads = 2,
                                    contextSize = null,
                                    storeChats = false,
                                    temperature = 0.0f,
                                    thinkingMode = SmolLM.ThinkingMode.DISABLED,
                                ),
                        )

                        onStatus?.invoke("Encoding image")
                        val projector = Projector()
                        projector.init(projectorFile.absolutePath, smol.getNativeModelPointer())
                        val encoded = projector.encodeImageToFile(imageFile.absolutePath, embedFile.absolutePath)
                        projector.close()

                        if (!encoded || !metaFile.exists()) {
                            imageFile.copyTo(embedFile, overwrite = true)
                        }

                        onStatus?.invoke("Reloading vision model")
                        adapter.loadVisionModel(
                            modelFile.absolutePath,
                            null,
                            SmolLM.InferenceParams(
                                numThreads = 2,
                                contextSize = 4096L,
                                storeChats = false,
                                temperature = 0.6f,
                                thinkingMode = SmolLM.ThinkingMode.DISABLED,
                            ),
                        )

                        onStatus?.invoke("Running vision analysis")
                        adapter
                            .analyze(
                                ImageSource.FileSource(embedFile),
                                request.prompt,
                                VisionParams(),
                            ).text
                    } finally {
                        if (metaFile.exists()) {
                            metaFile.delete()
                        }
                        if (embedFile.exists()) {
                            embedFile.delete()
                        }
                        if (imageFile.exists()) {
                            imageFile.delete()
                        }
                    }
                } finally {
                    adapter.close()
                }
            } finally {
                smol.close()
            }
        }
}
