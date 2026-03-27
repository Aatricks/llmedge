package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.text.runtime.SmolLM
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
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

    private data class RuntimeHandle(
        val cacheKey: VisionRuntimeCache.CacheKey,
        val smol: SmolLM,
        val projector: Projector,
        val isWarm: Boolean,
    )

    private val runtimeCache = VisionRuntimeCache(maxEntries = 1)
    private val runtimeMutex = Mutex()

    suspend fun prepare(
        model: ModelSpec,
        projector: ModelSpec,
        numThreads: Int,
        generationThreads: Int,
        onStatus: ((String) -> Unit)? = null,
    ) {
        withContext(Dispatchers.IO) {
            val runtime =
                acquireRuntime(
                    model = model,
                    projector = projector,
                    numThreads = numThreads,
                    generationThreads = generationThreads,
                    onStatus = onStatus,
                    cacheOnCold = true,
                )
            AndroidLogAdapter.d(
                TAG,
                "prepare completed for ${runtime.cacheKey.modelPath} (warm=${runtime.isWarm})",
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
                    onStatus = onStatus,
                )
            val smol = runtime.smol
            val projector = runtime.projector
            val cacheKey = runtime.cacheKey
            if (runtime.isWarm) {
                // Clear KV cache for a fresh vision prompt while keeping the model loaded.
                smol.clearKvCache()
            }

            try {
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

                        if (!runtime.isWarm) {
                            runtimeCache.put(cacheKey, VisionRuntimeCache.CachedRuntime(smol, projector))
                        }

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
                                "Buffer-based embedding decode failed for ${File(cacheKey.projectorPath).name}."
                            }

                            val generationStartedNs = System.nanoTime()
                            val response = smol.getResponse(
                                query = request.prompt,
                                batchSize = SmolLM.DEFAULT_BLOCKING_BATCH_SIZE,
                            )
                            logStage("analyze", "generation", generationStartedNs)

                            if (!runtime.isWarm) {
                                runtimeCache.put(cacheKey, VisionRuntimeCache.CachedRuntime(smol, projector))
                            }

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
                        "Native projector support is unavailable or failed to produce embeddings for ${File(cacheKey.projectorPath).name}."
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

                    if (!runtime.isWarm) {
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
                if (!runtime.isWarm) {
                    try { projector.close() } catch (_: Exception) {}
                    try { smol.close() } catch (_: Exception) {}
                }
                throw e
            }
        }

    private suspend fun acquireRuntime(
        model: ModelSpec,
        projector: ModelSpec,
        numThreads: Int?,
        generationThreads: Int?,
        onStatus: ((String) -> Unit)?,
        cacheOnCold: Boolean = false,
    ): RuntimeHandle =
        runtimeMutex.withLock {
            val resolveStartedNs = System.nanoTime()
            val modelFile = resolver.resolve(context, model)
            val projectorFile = resolver.resolve(context, projector)
            logStage("runtime", "resolve", resolveStartedNs)

            val cacheKey =
                VisionRuntimeCache.CacheKey(
                    modelPath = modelFile.absolutePath,
                    projectorPath = projectorFile.absolutePath,
                )
            val cached = runtimeCache.get(cacheKey)
            if (cached != null) {
                return@withLock RuntimeHandle(
                    cacheKey = cacheKey,
                    smol = cached.smolLM,
                    projector = cached.projector,
                    isWarm = true,
                )
            }

            onStatus?.invoke("Loading vision model")
            val loadStartedNs = System.nanoTime()
            val smol = smolLmFactory(config.textUseVulkan)
            val adapter = SmolLMVisionAdapter(context, smol)
            adapter.loadVisionModel(
                modelPath = modelFile.absolutePath,
                mmprojPath = projectorFile.absolutePath,
                params =
                    SmolLM.InferenceParams(
                        numThreads = (numThreads ?: config.defaultTextThreads).coerceAtLeast(1),
                        generationThreads =
                            (generationThreads
                                ?: numThreads
                                ?: config.defaultTextGenerationThreads).coerceAtLeast(1),
                        contextSize = null,
                        storeChats = false,
                        temperature = 0.0f,
                        useFlashAttn = config.defaultUseFlashAttention,
                        thinkingMode = SmolLM.ThinkingMode.DEFAULT,
                    ),
            )
            logStage("runtime", "load_model", loadStartedNs)

            val projectorLoadStartedNs = System.nanoTime()
            val runtimeProjector = projectorFactory()
            runtimeProjector.init(projectorFile.absolutePath, smol.getNativeModelPointer())
            check(runtimeProjector.isReady()) {
                "Native projector initialization failed for ${projectorFile.name}. Ensure the mmproj file matches the selected model and that projector bindings are available."
            }
            logStage("runtime", "init_projector", projectorLoadStartedNs)

            val runtime =
                RuntimeHandle(
                    cacheKey = cacheKey,
                    smol = smol,
                    projector = runtimeProjector,
                    isWarm = false,
                )
            if (cacheOnCold) {
                runtimeCache.put(cacheKey, VisionRuntimeCache.CachedRuntime(smol, runtimeProjector))
            }
            runtime
        }

    private fun logStage(operation: String, stage: String, startedNs: Long) {
        val elapsedMs = (System.nanoTime() - startedNs) / 1_000_000
        AndroidLogAdapter.d(TAG, "$operation.$stage completed in ${elapsedMs}ms")
    }

    override fun close() {
        runtimeCache.releaseAll()
    }
}
