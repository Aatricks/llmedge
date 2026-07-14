package io.aatricks.llmedge.image.diffusion

import android.content.Context
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeLibraryCatalog
import io.aatricks.llmedge.core.UnsupportedModelException
import io.aatricks.llmedge.core.runtime.RuntimeLoadPolicy
import io.aatricks.llmedge.core.runtime.runBackendAttempts
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.huggingface.WanModelEntry
import io.aatricks.llmedge.huggingface.WanModelRegistry
import io.aatricks.llmedge.runtime.ComputeBackend
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal data class StableDiffusionResolvedAssets(
    val modelPath: String,
    val vaePath: String?,
    val t5xxlPath: String?,
    val metadata: VideoModelMetadata,
    // Split-model (FLUX.2 Klein): when set, the DiT lives here and native modelPath is left empty.
    val diffusionModelPath: String? = null,
    val llmPath: String? = null,
    // Encoder-only (FLUX.2 / SD3 sequential precompute phase): load ONLY the text encoder(s), no DiT.
    val encoderOnly: Boolean = false,
    val componentPaths: StableDiffusionComponentPaths? = null,
)

internal object StableDiffusionLoadSupport {
    private const val LOG_TAG = "StableDiffusion"

    suspend fun resolveRequestedAssets(
        context: Context,
        request: StableDiffusionAssetRequest,
        validateResolvedAssets: (String, String?, String?, String?, String?, StableDiffusionComponentPaths?) -> Unit,
        inferVideoModelMetadata: suspend (String, String?, String?) -> VideoModelMetadata,
        onFallback: (String) -> Unit = {},
    ): StableDiffusionResolvedAssets =
        withContext(Dispatchers.IO) {
            // Encoder-only (FLUX.2 sequential precompute): only an llmPath is supplied. Load just the
            // Qwen3 encoder so the precompute phase peaks at the encoder size, not encoder+DiT.
            if (request.diffusionModelPath == null && request.modelPath == null && request.llmPath != null && (request.componentPaths == null || request.componentPaths.isAllNull())) {
                return@withContext StableDiffusionResolvedAssets(
                    modelPath = request.llmPath,
                    vaePath = null,
                    t5xxlPath = null,
                    llmPath = request.llmPath,
                    diffusionModelPath = null,
                    encoderOnly = true,
                    metadata = inferVideoModelMetadata(request.llmPath, request.modelId, request.filename),
                    componentPaths = null,
                )
            }

            // SD3 encoder-only sequential conditioning: either clipL and clipG are populated (CLIP-L/G-only phase) or t5xxl is populated (T5XXL-only phase).
            val componentPaths = request.componentPaths
            val isSd3EncoderOnly = request.modelPath == null &&
                request.diffusionModelPath == null &&
                request.llmPath == null &&
                ((request.t5xxlPath == null && componentPaths?.clipLPath != null && componentPaths.clipGPath != null) ||
                 (request.t5xxlPath != null && (componentPaths == null || (componentPaths.clipLPath == null && componentPaths.clipGPath == null))))

            if (isSd3EncoderOnly) {
                val metadataPath = componentPaths?.clipLPath ?: request.t5xxlPath!!
                return@withContext StableDiffusionResolvedAssets(
                    modelPath = metadataPath,
                    vaePath = null,
                    t5xxlPath = request.t5xxlPath,
                    llmPath = null,
                    diffusionModelPath = null,
                    encoderOnly = true,
                    metadata = inferVideoModelMetadata(metadataPath, request.modelId, request.filename),
                    componentPaths = componentPaths,
                )
            }

            // Split-model (FLUX.2 Klein): caller supplies pre-resolved absolute paths for the DiT,
            // Qwen3 encoder and VAE. The DiT path doubles as modelPath for metadata/heuristics, but
            // the native layer routes it to diffusion_model_path and leaves model_path empty.
            if (request.diffusionModelPath != null) {
                validateResolvedAssets(
                    request.diffusionModelPath,
                    request.vaePath,
                    request.t5xxlPath,
                    request.taesdPath,
                    request.loraModelDir,
                    request.componentPaths,
                )
                return@withContext StableDiffusionResolvedAssets(
                    modelPath = request.diffusionModelPath,
                    vaePath = request.vaePath,
                    t5xxlPath = request.t5xxlPath,
                    diffusionModelPath = request.diffusionModelPath,
                    llmPath = request.llmPath,
                    metadata =
                        inferVideoModelMetadata(
                            request.diffusionModelPath,
                            request.modelId,
                            request.filename,
                        ),
                    componentPaths = request.componentPaths,
                )
            }

            if (request.modelPath != null) {
                validateResolvedAssets(
                    request.modelPath,
                    request.vaePath,
                    request.t5xxlPath,
                    request.taesdPath,
                    request.loraModelDir,
                    request.componentPaths,
                )
                return@withContext StableDiffusionResolvedAssets(
                    modelPath = request.modelPath,
                    vaePath = request.vaePath,
                    t5xxlPath = request.t5xxlPath,
                    encoderOnly = request.componentPaths?.miniT2iConditionerOnly == true,
                    metadata =
                        inferVideoModelMetadata(
                            request.modelPath,
                            request.modelId,
                            request.filename,
                        ),
                    componentPaths = request.componentPaths,
                )
            }

            val resolvedModelId = request.modelId ?: throw io.aatricks.llmedge.core.InvalidGenerationParametersException(
                "Provide either modelPath or modelId",
            )

            val possibleWan =
                WanModelRegistry.findById(context, resolvedModelId)
                    ?: WanModelRegistry.findByModelIdPrefix(
                        context,
                        resolvedModelId.removePrefix("wan/"),
                    )
            if (possibleWan != null) {
                return@withContext resolveWanAssets(
                    context = context,
                    request =
                        request.copy(
                            modelId = resolvedModelId,
                            preferSystemDownloader = true,
                        ),
                    onProgress = null,
                    validateResolvedAssets = validateResolvedAssets,
                    inferVideoModelMetadata = inferVideoModelMetadata,
                    registryEntry = possibleWan,
                )
            }

            val resolvedModelPath =
                try {
                    HuggingFaceHub.ensureModelOnDisk(
                        context = context,
                        modelId = resolvedModelId,
                        revision = "main",
                        preferredQuantizations = emptyList(),
                        filename = request.filename,
                        token = request.token,
                        forceDownload = request.forceDownload,
                        preferSystemDownloader = request.preferSystemDownloader,
                        onProgress = null,
                    ).file.absolutePath
                } catch (error: IllegalArgumentException) {
                    onFallback(
                        "Falling back to generic repo-file resolution for $resolvedModelId${request.filename?.let { " ($it)" } ?: ""}: ${error.message}",
                    )
                    HuggingFaceHub.ensureRepoFileOnDisk(
                        context = context,
                        modelId = resolvedModelId,
                        revision = "main",
                        filename = request.filename,
                        allowedExtensions = listOf(".gguf", ".safetensors", ".ckpt", ".pt", ".bin"),
                        token = request.token,
                        forceDownload = request.forceDownload,
                        preferSystemDownloader = request.preferSystemDownloader,
                        onProgress = null,
                    ).file.absolutePath
                }

            validateResolvedAssets(
                resolvedModelPath,
                request.vaePath,
                request.t5xxlPath,
                request.taesdPath,
                request.loraModelDir,
                request.componentPaths,
            )

            StableDiffusionResolvedAssets(
                modelPath = resolvedModelPath,
                vaePath = request.vaePath,
                t5xxlPath = request.t5xxlPath,
                metadata = inferVideoModelMetadata(resolvedModelPath, resolvedModelId, request.filename),
                componentPaths = request.componentPaths,
            )
        }

    suspend fun resolveWanAssets(
        context: Context,
        request: StableDiffusionAssetRequest,
        onProgress: ((name: String, downloaded: Long, total: Long?) -> Unit)?,
        validateResolvedAssets: (String, String?, String?, String?, String?, StableDiffusionComponentPaths?) -> Unit,
        inferVideoModelMetadata: suspend (String, String?, String?) -> VideoModelMetadata,
        registryEntry: WanModelEntry? = null,
    ): StableDiffusionResolvedAssets =
        withContext(Dispatchers.IO) {
            val modelId =
                requireNotNull(request.modelId) {
                    "request.modelId is required when resolving Wan assets."
                }
            val (modelRes, vaeRes, t5Res) =
                HuggingFaceHub.ensureWanAssetsOnDisk(
                    context = context,
                    wanModelId = modelId,
                    preferSystemDownloader = request.preferSystemDownloader,
                    token = request.token,
                    forceDownload = request.forceDownload,
                    onProgress = { downloaded, total ->
                        onProgress?.invoke(modelId, downloaded, total)
                    },
                )

            val resolvedModelPath = modelRes.file.absolutePath
            val resolvedVaePath = vaeRes?.file?.absolutePath
            val resolvedT5xxlPath = t5Res?.file?.absolutePath

            validateResolvedAssets(
                resolvedModelPath,
                resolvedVaePath,
                resolvedT5xxlPath,
                request.taesdPath,
                request.loraModelDir,
                request.componentPaths,
            )

            StableDiffusionResolvedAssets(
                modelPath = resolvedModelPath,
                vaePath = resolvedVaePath,
                t5xxlPath = resolvedT5xxlPath,
                metadata =
                    registryEntry?.toVideoModelMetadata(resolvedModelPath.substringAfterLast('/'))
                        ?: inferVideoModelMetadata(resolvedModelPath, modelId, request.filename),
                componentPaths = request.componentPaths,
            )
        }

    suspend fun inferVideoModelMetadata(
        resolvedModelPath: String,
        modelId: String?,
        explicitFilename: String?,
    ): VideoModelMetadata =
        StableDiffusionMetadataSupport.inferVideoModelMetadata(
            resolvedModelPath = resolvedModelPath,
            modelId = modelId,
            explicitFilename = explicitFilename,
        )

    fun validateResolvedAssets(
        modelPath: String,
        vaePath: String?,
        t5xxlPath: String?,
        taesdPath: String?,
        loraModelDir: String?,
        componentPaths: StableDiffusionComponentPaths? = null,
    ) {
        StableDiffusionLoadHeuristics.validateResolvedAssets(
            modelPath = modelPath,
            vaePath = vaePath,
            t5xxlPath = t5xxlPath,
            taesdPath = taesdPath,
            loraModelDir = loraModelDir,
            componentPaths = componentPaths,
        )
    }

    fun logLoadFallback(message: String) {
        AndroidLogAdapter.w(LOG_TAG, message)
    }

    fun createLoadedInstance(
        context: Context,
        resolved: StableDiffusionResolvedAssets,
        request: StableDiffusionLoadRequest,
    ): StableDiffusion {
        val loadPlan =
            StableDiffusionLoadHeuristics.planLoad(
                context = context,
                resolvedModelPath = resolved.modelPath,
                sequentialLoad = request.runtime.sequentialLoad,
                preferPerformanceMode = request.runtime.preferPerformanceMode,
                offloadToCpu = request.runtime.offloadToCpu,
                keepClipOnCpu = request.runtime.keepClipOnCpu,
                keepVaeOnCpu = request.runtime.keepVaeOnCpu,
                allowOpenCl = request.backend.allowOpenCl,
                allowVulkan = request.backend.allowVulkan,
                forceVulkan = request.backend.forceVulkan,
            )
        StableDiffusionLoadHeuristics.warnIfLargeModelOnLowRam(
            metadata = resolved.metadata,
            memorySnapshot = loadPlan.memorySnapshot,
        ) { message -> AndroidLogAdapter.w(LOG_TAG, message) }

        logLoadPlan(
            resolvedModelPath = resolved.modelPath,
            nThreads = request.runtime.nThreads,
            loadPlan = loadPlan,
            flashAttn = request.runtime.flashAttn,
        )

        val handle =
            createHandleWithBackendFallback(
                resolved = resolved,
                request = request,
                loadPlan = loadPlan,
                allowBackendFallbackToCpu = request.backend.allowBackendFallbackToCpu,
            )
        if (handle == 0L) {
            throw ModelLoadException(
                resolved.modelPath,
                createLoadFailureMessage(
                    resolvedModelPath = resolved.modelPath,
                    taesdPath = request.assets.taesdPath,
                    resolvedVaePath = resolved.vaePath,
                ),
            )
        }

        val instance = StableDiffusion(handle)
        instance.state.vulkanEnabledForMetrics = loadPlan.chosenBackend == ComputeBackend.VULKAN
        instance.updateModelMetadata(resolved.metadata)

        if (instance.state.modelMetadata?.mobileSupported == false) {
            val paramCount = instance.state.modelMetadata?.parameterCount ?: "14B"
            instance.close()
            throw UnsupportedModelException(
                "$paramCount models are not supported on mobile devices. " +
                    "Please use 1.3B or 5B model variants instead. " +
                    "14B models require 20-40GB RAM and are designed for desktop/server use only.",
            )
        }

        return instance
    }

    private fun logLoadPlan(
        resolvedModelPath: String,
        nThreads: Int,
        loadPlan: StableDiffusionLoadHeuristics.LoadPlan,
        flashAttn: Boolean,
    ) {
        AndroidLogAdapter.i(
            LOG_TAG,
            "Initializing StableDiffusion (effective): modelPath=$resolvedModelPath, " +
                "nThreads=$nThreads, sequentialLoad=${loadPlan.effectiveSequentialLoad}, " +
                "offloadToCpu=${loadPlan.effectiveOffloadToCpu}, " +
                "keepClipOnCpu=${loadPlan.effectiveKeepClipOnCpu}, backend=${loadPlan.chosenBackend}, " +
                "keepVaeOnCpu=${loadPlan.effectiveKeepVaeOnCpu}, flashAttn=$flashAttn",
        )
        if (loadPlan.chosenDevice >= 0) {
            AndroidLogAdapter.i(
                LOG_TAG,
                "Vulkan chosenDevice=${loadPlan.chosenDevice}, estimatedModelParamsMB=${String.format("%.2f", loadPlan.estimatedDeviceParamsBytes / 1024.0 / 1024.0)}, freeMB=${String.format("%.2f", loadPlan.freeVulkanBytes / 1024.0 / 1024.0)}",
            )
        }
    }

    private fun createHandleWithBackendFallback(
        resolved: StableDiffusionResolvedAssets,
        request: StableDiffusionLoadRequest,
        loadPlan: StableDiffusionLoadHeuristics.LoadPlan,
        allowBackendFallbackToCpu: Boolean,
    ): Long {
        var effectiveOffloadToCpu = loadPlan.effectiveOffloadToCpu
        var effectiveKeepClipOnCpu = loadPlan.effectiveKeepClipOnCpu
        var effectiveKeepVaeOnCpu = loadPlan.effectiveKeepVaeOnCpu

        var handle =
            runBackendAttempts(
                candidates = RuntimeLoadPolicy.candidates(loadPlan.chosenBackend, allowBackendFallbackToCpu),
                onFailure = { backend, error ->
                    if (backend != ComputeBackend.CPU) {
                        val detail = error?.message?.let { ": $it" } ?: ""
                        AndroidLogAdapter.w(LOG_TAG, "nativeCreate failed on $backend; retrying with CPU backend$detail")
                    }
                },
            ) { backend ->
                nativeCreateOrThrow(
                    createNativeLoadRequest(
                        resolved = resolved,
                        request = request,
                        backend = backend,
                        offloadToCpu = effectiveOffloadToCpu,
                        keepClipOnCpu = effectiveKeepClipOnCpu,
                        keepVaeOnCpu = effectiveKeepVaeOnCpu,
                    ),
                ).takeIf { it != 0L }
            } ?: 0L
        if (handle == 0L && !effectiveOffloadToCpu) {
            AndroidLogAdapter.w(LOG_TAG, "nativeCreate failed on CPU backend; retrying with CPU offload")
            effectiveOffloadToCpu = true
            effectiveKeepClipOnCpu = true
            effectiveKeepVaeOnCpu = true
            handle =
                nativeCreateOrThrow(
                    createNativeLoadRequest(
                        resolved = resolved,
                        request = request,
                        backend = ComputeBackend.CPU,
                        offloadToCpu = effectiveOffloadToCpu,
                        keepClipOnCpu = effectiveKeepClipOnCpu,
                        keepVaeOnCpu = effectiveKeepVaeOnCpu,
                    ),
                )
        }
        return handle
    }

    private fun createNativeLoadRequest(
        resolved: StableDiffusionResolvedAssets,
        request: StableDiffusionLoadRequest,
        backend: ComputeBackend,
        offloadToCpu: Boolean,
        keepClipOnCpu: Boolean,
        keepVaeOnCpu: Boolean,
    ): StableDiffusionNativeLoadRequest =
        StableDiffusionNativeLoadRequest(
            // For split models the DiT must go to diffusion_model_path; model_path must be empty
            // so sdcpp doesn't try to load it as a complete checkpoint. Encoder-only loads also
            // leave model_path empty (routing text encoders appropriately).
            modelPath =
                if (resolved.diffusionModelPath != null ||
                    (resolved.encoderOnly && resolved.componentPaths?.miniT2iConditionerOnly != true)
                ) {
                    ""
                } else {
                    resolved.modelPath
                },
            vaePath = resolved.vaePath,
            t5xxlPath = resolved.t5xxlPath,
            taesdPath = request.assets.taesdPath,
            diffusionModelPath = resolved.diffusionModelPath,
            llmPath = resolved.llmPath,
            clipLPath = resolved.componentPaths?.clipLPath,
            clipGPath = resolved.componentPaths?.clipGPath,
            clipVisionPath = resolved.componentPaths?.clipVisionPath,
            llmVisionPath = resolved.componentPaths?.llmVisionPath,
            highNoiseDiffusionModelPath = resolved.componentPaths?.highNoiseDiffusionModelPath,
            embeddingsConnectorsPath = resolved.componentPaths?.embeddingsConnectorsPath,
            audioVaePath = resolved.componentPaths?.audioVaePath,
            controlNetPath = resolved.componentPaths?.controlNetPath,
            photoMakerPath = resolved.componentPaths?.photoMakerPath,
            nThreads = request.runtime.nThreads,
            enableOpenCl = backend == ComputeBackend.OPENCL,
            useVulkan = backend == ComputeBackend.VULKAN,
            offloadToCpu = offloadToCpu,
            keepClipOnCpu = keepClipOnCpu,
            keepVaeOnCpu = keepVaeOnCpu,
            flashAttn = request.runtime.flashAttn,
            vaeDecodeOnly = request.runtime.vaeDecodeOnly,
            flowShift = request.runtime.flowShift,
            loraModelDir = request.assets.loraModelDir,
            loraApplyMode = request.runtime.loraApplyMode,
            miniT2iConditionerOnly = resolved.componentPaths?.miniT2iConditionerOnly == true,
        )

    private fun nativeCreateOrThrow(
        request: StableDiffusionNativeLoadRequest,
    ): Long =
        try {
            StableDiffusion.supportNativeCreate(request)
        } catch (error: UnsatisfiedLinkError) {
            throw NativeBindingException(
                libraryName = NativeLibraryCatalog.STABLE_DIFFUSION,
                detail = "Stable Diffusion JNI bindings are unavailable.",
                cause = error,
            )
        }

    private fun createLoadFailureMessage(
        resolvedModelPath: String,
        taesdPath: String?,
        resolvedVaePath: String?,
    ): String =
        buildString {
            append("Failed to initialize Stable Diffusion context for $resolvedModelPath.")
            if (taesdPath != null) append(" Custom TAE/TAEHV: $taesdPath.")
            if (resolvedVaePath != null) append(" Custom VAE: $resolvedVaePath.")
            append(" This often happens due to incompatible VAE/TAE weights or insufficient memory. Check logcat for [SmolSD] errors.")
        }
}

private fun WanModelEntry.toVideoModelMetadata(
    resolvedFilename: String,
): VideoModelMetadata =
    VideoModelMetadata(
        architecture = architecture ?: modelId,
        modelType = modelType,
        parameterCount = description?.let(::extractParameterCount),
        mobileSupported = mobileSupported,
        tags =
            buildSet {
                add("wan")
                modelType?.let { add(it) }
                quantization?.let { add(it) }
            },
        filename = resolvedFilename,
    )

private fun extractParameterCount(description: String): String? {
    val normalized = description.replace(" ", "")
    return listOf("1.3B", "5B", "14B").firstOrNull { normalized.contains(it, ignoreCase = true) }
}
