package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

internal class VisionPipeline(
    private val context: Context,
    private val resolver: ModelResolver,
    private val config: LLMEdgeConfig,
    private val smolLmFactory: (Boolean) -> SmolLM = { useVulkan -> SmolLM(useVulkan = useVulkan) },
    private val projectorFactory: () -> Projector = { Projector() },
) : AutoCloseable {
    private companion object {
        private const val TAG = "VisionPipeline"
        private const val JPEG_QUALITY = 90
    }

    internal data class VisionPipelineResult(
        val text: String,
        val runtimeMemory: VisionRuntimeMemory,
    )

    private val runtimePool =
        createVisionRuntimePool(
            context = context,
            resolver = resolver,
            config = config,
            smolLmFactory = smolLmFactory,
            projectorFactory = projectorFactory,
        )

    suspend fun prepare(
        model: ModelSpec,
        projector: ModelSpec,
        numThreads: Int,
        generationThreads: Int,
        onStatus: ((String) -> Unit)? = null,
    ) {
        withContext(Dispatchers.IO) {
            acquireRuntime(
                model = model,
                projector = projector,
                numThreads = numThreads,
                generationThreads = generationThreads,
            )
            AndroidLogAdapter.d(
                TAG,
                "prepare completed for ${model.cacheKey}",
            )
        }
    }

    suspend fun analyze(
        request: VisionRequest,
        onStatus: ((String) -> Unit)? = null,
    ): VisionPipelineResult =
        withContext(Dispatchers.IO) {
            val runtime =
                acquireRuntime(
                    model = request.model,
                    projector = request.projector,
                    numThreads = request.numThreads,
                    generationThreads = request.generationThreads,
                )
            runtime.mutex.withLock {
                val smol = runtime.smol
                val projector = runtime.projector
                smol.clearKvCache()
                onStatus?.invoke("Preparing image")
                val imagePrepStartedNs = System.nanoTime()
                val scaled =
                    ImageUtils.preprocessBitmap(
                        request.image,
                        maxDimension = 672,
                        enhance = false,
                    )
                logStage("analyze", "image_prep", imagePrepStartedNs)

                // Try buffer-based path first (zero disk I/O)
                val bufferSuccess = try {
                    val jpegStream = java.io.ByteArrayOutputStream()
                    scaled.compress(android.graphics.Bitmap.CompressFormat.JPEG, JPEG_QUALITY, jpegStream)
                    val jpegBytes = jpegStream.toByteArray()

                    onStatus?.invoke("Preparing multimodal embeddings")
                    val embeddingStartedNs = System.nanoTime()
                    val primed = smol.primeImageBuffer(projector.nativeHandle(), jpegBytes, nBatch = 1)
                    logStage("analyze", "prime_image_buffer", embeddingStartedNs)
                    if (primed) {
                        onStatus?.invoke("Running vision analysis")
                        val generationStartedNs = System.nanoTime()
                        val response = smol.getResponse(
                            query = request.prompt,
                            batchSize = SmolLM.DEFAULT_BLOCKING_BATCH_SIZE,
                        )
                        logStage("analyze", "generation", generationStartedNs)

                        VisionPipelineResult(
                            text = response,
                            runtimeMemory = VisionRuntimeMemory(
                                nativeBytes = smol.getEstimatedNativeMemoryBytes(),
                                stateBytes = smol.getEstimatedStateMemoryBytes(),
                            ),
                        )
                    } else {
                        val embeddingStartedNs = System.nanoTime()
                        val embeddings = projector.encodeImageBuffer(jpegBytes)
                        logStage("analyze", "encode_image_buffer", embeddingStartedNs)
                        if (embeddings != null) {
                            onStatus?.invoke("Running vision analysis")
                            val decodeStartedNs = System.nanoTime()
                            val decodeOk = smol.decodeEmbeddingsBuffer(embeddings, nBatch = 1)
                            logStage("analyze", "decode_embeddings_buffer", decodeStartedNs)
                            check(decodeOk) {
                                "Buffer-based embedding decode failed for the active vision runtime."
                            }

                            val generationStartedNs = System.nanoTime()
                            val response = smol.getResponse(
                                query = request.prompt,
                                batchSize = SmolLM.DEFAULT_BLOCKING_BATCH_SIZE,
                            )
                            logStage("analyze", "generation", generationStartedNs)

                            VisionPipelineResult(
                                text = response,
                                runtimeMemory = VisionRuntimeMemory(
                                    nativeBytes = smol.getEstimatedNativeMemoryBytes(),
                                    stateBytes = smol.getEstimatedStateMemoryBytes(),
                                ),
                            )
                        } else {
                            null
                        }
                    }
                } catch (_: UnsatisfiedLinkError) {
                    null
                }

                if (bufferSuccess != null) {
                    return@withContext bufferSuccess
                }

                // Fallback: file-based path
                val imageFile = File.createTempFile("vision_input", ".jpg", context.cacheDir)
                val embedFile = File.createTempFile("vision_prepared", ".bin", context.cacheDir)
                val metaFile = File(embedFile.absolutePath + ".meta.json")

                try {
                    imageFile.outputStream().use { out ->
                        scaled.compress(android.graphics.Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
                    }

                    onStatus?.invoke("Preparing multimodal embeddings")
                    val embeddingStartedNs = System.nanoTime()
                    val encoded = projector.encodeImageToFile(imageFile.absolutePath, embedFile.absolutePath)
                    logStage("analyze", "encode_image_file", embeddingStartedNs)
                    check(encoded && metaFile.exists()) {
                        "Native projector support is unavailable or failed to produce embeddings."
                    }

                    onStatus?.invoke("Running vision analysis")
                    val adapter = SmolLMVisionAdapter(context, smol)
                    val generationStartedNs = System.nanoTime()
                    val analysis =
                        adapter.analyze(
                            ImageSource.FileSource(embedFile),
                            request.prompt,
                            VisionParams(),
                        )
                    logStage("analyze", "generation", generationStartedNs)

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
            }
        }

    private suspend fun acquireRuntime(
        model: ModelSpec,
        projector: ModelSpec,
        numThreads: Int?,
        generationThreads: Int?,
    ): ManagedVisionRuntime {
        val loadStartedNs = System.nanoTime()
        val runtime =
            runtimePool.acquire(
                VisionRuntimeSpec(model = model, projector = projector),
                VisionLoadOptions(
                    numThreads = (numThreads ?: config.text.promptThreads).coerceAtLeast(1),
                    generationThreads = (generationThreads ?: numThreads ?: config.text.generationThreads).coerceAtLeast(1),
                ),
            )
        logStage("runtime", "acquire", loadStartedNs)
        return runtime
    }

    private fun logStage(operation: String, stage: String, startedNs: Long) {
        val elapsedMs = (System.nanoTime() - startedNs) / 1_000_000
        AndroidLogAdapter.d(TAG, "$operation.$stage completed in ${elapsedMs}ms")
    }

    override fun close() {
        runtimePool.close()
    }
}
