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
) : AutoCloseable {
    internal data class VisionPipelineResult(
        val text: String,
        val runtimeMemory: VisionRuntimeMemory,
    )

    private val runtimeCache = VisionRuntimeCache(maxEntries = 1)

    suspend fun analyze(
        request: VisionRequest,
        onStatus: ((String) -> Unit)? = null,
    ): VisionPipelineResult =
        withContext(Dispatchers.IO) {
            val modelFile = resolver.resolve(context, request.model)
            val projectorFile = resolver.resolve(context, request.projector)
            val cacheKey = VisionRuntimeCache.CacheKey(
                modelPath = modelFile.absolutePath,
                projectorPath = projectorFile.absolutePath,
            )

            val cached = runtimeCache.get(cacheKey)
            val smol: SmolLM
            val projector: Projector
            val isWarm: Boolean

            if (cached != null) {
                smol = cached.smolLM
                projector = cached.projector
                isWarm = true
                // Clear KV cache for a fresh vision prompt while keeping the model loaded
                smol.clearKvCache()
            } else {
                onStatus?.invoke("Loading vision model")
                smol = SmolLM(useVulkan = false)
                val adapter = SmolLMVisionAdapter(context, smol)
                adapter.loadVisionModel(
                    modelPath = modelFile.absolutePath,
                    mmprojPath = projectorFile.absolutePath,
                    params =
                        SmolLM.InferenceParams(
                            numThreads = request.numThreads?.coerceAtLeast(1) ?: 2,
                            generationThreads =
                                request.generationThreads?.coerceAtLeast(1)
                                    ?: request.numThreads?.coerceAtLeast(1)
                                    ?: 2,
                            contextSize = null,
                            storeChats = false,
                            temperature = 0.0f,
                            useFlashAttn = false,
                            thinkingMode = SmolLM.ThinkingMode.DEFAULT,
                        ),
                )
                projector = Projector()
                projector.init(projectorFile.absolutePath, smol.getNativeModelPointer())
                check(projector.isReady()) {
                    "Native projector initialization failed for ${projectorFile.name}. Ensure the mmproj file matches the selected model and that projector bindings are available."
                }
                isWarm = false
            }

            try {
                val imageFile = File.createTempFile("vision_input", ".jpg", context.cacheDir)
                val embedFile = File.createTempFile("vision_prepared", ".bin", context.cacheDir)
                val metaFile = File(embedFile.absolutePath + ".meta.json")

                try {
                    onStatus?.invoke("Preparing image")
                    val scaled =
                        ImageUtils.preprocessBitmap(
                            request.image,
                            maxDimension = 672,
                            enhance = false,
                        )
                    imageFile.outputStream().use { out ->
                        scaled.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
                    }

                    onStatus?.invoke("Preparing multimodal embeddings")
                    val encoded = projector.encodeImageToFile(imageFile.absolutePath, embedFile.absolutePath)
                    check(encoded && metaFile.exists()) {
                        "Native projector support is unavailable or failed to produce embeddings for ${projectorFile.name}."
                    }

                    onStatus?.invoke("Running vision analysis")
                    val adapter = SmolLMVisionAdapter(context, smol)
                    val analysis =
                        adapter.analyze(
                            ImageSource.FileSource(embedFile),
                            request.prompt,
                            VisionParams(),
                        )

                    // Cache the runtime for reuse on subsequent calls
                    if (!isWarm) {
                        runtimeCache.put(cacheKey, VisionRuntimeCache.CachedRuntime(smol, projector))
                    }

                    VisionPipelineResult(
                        text = analysis.text,
                        runtimeMemory =
                            VisionRuntimeMemory(
                                nativeBytes = smol.getEstimatedNativeMemoryBytes(),
                                stateBytes = smol.getEstimatedStateMemoryBytes(),
                            ),
                    )
                } finally {
                    if (metaFile.exists()) metaFile.delete()
                    if (embedFile.exists()) embedFile.delete()
                    if (imageFile.exists()) imageFile.delete()
                }
            } catch (e: Exception) {
                // On failure with a freshly created runtime, clean up to avoid leaks
                if (!isWarm) {
                    try { projector.close() } catch (_: Exception) {}
                    try { smol.close() } catch (_: Exception) {}
                }
                throw e
            }
        }

    override fun close() {
        runtimeCache.releaseAll()
    }
}
