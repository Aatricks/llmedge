package io.aatricks.llmedge.image.diffusion

import android.content.Context
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeLibraryCatalog
import io.aatricks.llmedge.core.UnsupportedModelException
import io.aatricks.llmedge.core.runtime.RuntimeLoadPolicy
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
)

internal object StableDiffusionLoadSupport {
    private const val LOG_TAG = "StableDiffusion"

    suspend fun resolveRequestedAssets(
        context: Context,
        modelId: String?,
        filename: String?,
        modelPath: String?,
        vaePath: String?,
        t5xxlPath: String?,
        taesdPath: String?,
        token: String?,
        forceDownload: Boolean,
        loraModelDir: String?,
        validateResolvedAssets: (String, String?, String?, String?, String?) -> Unit,
        inferVideoModelMetadata: suspend (String, String?, String?) -> VideoModelMetadata,
        onFallback: (String) -> Unit = {},
    ): StableDiffusionResolvedAssets =
        withContext(Dispatchers.IO) {
            if (modelPath != null) {
                validateResolvedAssets(
                    modelPath,
                    vaePath,
                    t5xxlPath,
                    taesdPath,
                    loraModelDir,
                )
                return@withContext StableDiffusionResolvedAssets(
                    modelPath = modelPath,
                    vaePath = vaePath,
                    t5xxlPath = t5xxlPath,
                    metadata = inferVideoModelMetadata(modelPath, modelId, filename),
                )
            }

            val resolvedModelId = modelId ?: throw io.aatricks.llmedge.core.InvalidGenerationParametersException(
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
                    modelId = resolvedModelId,
                    filename = filename,
                    taesdPath = taesdPath,
                    token = token,
                    forceDownload = forceDownload,
                    preferSystemDownloader = true,
                    loraModelDir = loraModelDir,
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
                        filename = filename,
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = true,
                        onProgress = null,
                    ).file.absolutePath
                } catch (error: IllegalArgumentException) {
                    onFallback(
                        "Falling back to generic repo-file resolution for $resolvedModelId${filename?.let { " ($it)" } ?: ""}: ${error.message}",
                    )
                    HuggingFaceHub.ensureRepoFileOnDisk(
                        context = context,
                        modelId = resolvedModelId,
                        revision = "main",
                        filename = filename,
                        allowedExtensions = listOf(".gguf", ".safetensors", ".ckpt", ".pt", ".bin"),
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = true,
                        onProgress = null,
                    ).file.absolutePath
                }

            validateResolvedAssets(
                resolvedModelPath,
                vaePath,
                t5xxlPath,
                taesdPath,
                loraModelDir,
            )

            StableDiffusionResolvedAssets(
                modelPath = resolvedModelPath,
                vaePath = vaePath,
                t5xxlPath = t5xxlPath,
                metadata = inferVideoModelMetadata(resolvedModelPath, resolvedModelId, filename),
            )
        }

    suspend fun resolveWanAssets(
        context: Context,
        modelId: String,
        filename: String?,
        taesdPath: String?,
        token: String?,
        forceDownload: Boolean,
        preferSystemDownloader: Boolean,
        loraModelDir: String?,
        onProgress: ((name: String, downloaded: Long, total: Long?) -> Unit)?,
        validateResolvedAssets: (String, String?, String?, String?, String?) -> Unit,
        inferVideoModelMetadata: suspend (String, String?, String?) -> VideoModelMetadata,
        registryEntry: WanModelEntry? = null,
    ): StableDiffusionResolvedAssets =
        withContext(Dispatchers.IO) {
            val (modelRes, vaeRes, t5Res) =
                HuggingFaceHub.ensureWanAssetsOnDisk(
                    context = context,
                    wanModelId = modelId,
                    preferSystemDownloader = preferSystemDownloader,
                    token = token,
                    forceDownload = forceDownload,
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
                taesdPath,
                loraModelDir,
            )

            StableDiffusionResolvedAssets(
                modelPath = resolvedModelPath,
                vaePath = resolvedVaePath,
                t5xxlPath = resolvedT5xxlPath,
                metadata =
                    registryEntry?.toVideoModelMetadata(resolvedModelPath.substringAfterLast('/'))
                        ?: inferVideoModelMetadata(resolvedModelPath, modelId, filename),
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
    ) {
        StableDiffusionLoadHeuristics.validateResolvedAssets(
            modelPath = modelPath,
            vaePath = vaePath,
            t5xxlPath = t5xxlPath,
            taesdPath = taesdPath,
            loraModelDir = loraModelDir,
        )
    }

    fun logLoadFallback(message: String) {
        AndroidLogAdapter.w(LOG_TAG, message)
    }

    fun createLoadedInstance(
        context: Context,
        resolved: StableDiffusionResolvedAssets,
        taesdPath: String?,
        nThreads: Int,
        offloadToCpu: Boolean,
        keepClipOnCpu: Boolean,
        keepVaeOnCpu: Boolean,
        flashAttn: Boolean,
        vaeDecodeOnly: Boolean,
        sequentialLoad: Boolean?,
        allowOpenCl: Boolean,
        allowVulkan: Boolean,
        forceVulkan: Boolean,
        preferPerformanceMode: Boolean,
        flowShift: Float,
        loraModelDir: String?,
        loraApplyMode: LoraApplyMode,
        preferredBackend: ComputeBackend?,
        allowBackendFallbackToCpu: Boolean,
    ): StableDiffusion {
        val loadPlan =
            StableDiffusionLoadHeuristics.planLoad(
                context = context,
                resolvedModelPath = resolved.modelPath,
                sequentialLoad = sequentialLoad,
                preferPerformanceMode = preferPerformanceMode,
                offloadToCpu = offloadToCpu,
                keepClipOnCpu = keepClipOnCpu,
                keepVaeOnCpu = keepVaeOnCpu,
                allowOpenCl = allowOpenCl,
                allowVulkan = allowVulkan,
                forceVulkan = forceVulkan,
            )
        StableDiffusionLoadHeuristics.warnIfLargeModelOnLowRam(
            metadata = resolved.metadata,
            memorySnapshot = loadPlan.memorySnapshot,
        ) { message -> AndroidLogAdapter.w(LOG_TAG, message) }

        logLoadPlan(
            resolvedModelPath = resolved.modelPath,
            nThreads = nThreads,
            loadPlan = loadPlan,
            flashAttn = flashAttn,
        )

        val handle =
            createHandleWithBackendFallback(
                resolved = resolved,
                taesdPath = taesdPath,
                nThreads = nThreads,
                loadPlan = loadPlan,
                flashAttn = flashAttn,
                vaeDecodeOnly = vaeDecodeOnly,
                flowShift = flowShift,
                loraModelDir = loraModelDir,
                loraApplyMode = loraApplyMode,
                allowBackendFallbackToCpu = allowBackendFallbackToCpu,
            )
        if (handle == 0L) {
            throw ModelLoadException(
                resolved.modelPath,
                createLoadFailureMessage(
                    resolvedModelPath = resolved.modelPath,
                    taesdPath = taesdPath,
                    resolvedVaePath = resolved.vaePath,
                ),
            )
        }

        val instance = StableDiffusion(handle)
        instance.state.vulkanEnabledForMetrics = loadPlan.chosenBackend == ComputeBackend.VULKAN
        instance.updateModelMetadata(resolved.metadata)

        if (instance.state.modelMetadata?.mobileSupported == false) {
            instance.close()
            val paramCount = instance.state.modelMetadata?.parameterCount ?: "14B"
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
        taesdPath: String?,
        nThreads: Int,
        loadPlan: StableDiffusionLoadHeuristics.LoadPlan,
        flashAttn: Boolean,
        vaeDecodeOnly: Boolean,
        flowShift: Float,
        loraModelDir: String?,
        loraApplyMode: LoraApplyMode,
        allowBackendFallbackToCpu: Boolean,
    ): Long {
        var effectiveOffloadToCpu = loadPlan.effectiveOffloadToCpu
        var effectiveKeepClipOnCpu = loadPlan.effectiveKeepClipOnCpu
        var effectiveKeepVaeOnCpu = loadPlan.effectiveKeepVaeOnCpu

        var handle = 0L
        for (backend in RuntimeLoadPolicy.candidates(loadPlan.chosenBackend, allowBackendFallbackToCpu)) {
            handle =
                nativeCreateOrThrow(
                    modelPath = resolved.modelPath,
                    vaePath = resolved.vaePath,
                    t5xxlPath = resolved.t5xxlPath,
                    taesdPath = taesdPath,
                    nThreads = nThreads,
                    enableOpenCl = backend == ComputeBackend.OPENCL,
                    useVulkan = backend == ComputeBackend.VULKAN,
                    offloadToCpu = effectiveOffloadToCpu,
                    keepClipOnCpu = effectiveKeepClipOnCpu,
                    keepVaeOnCpu = effectiveKeepVaeOnCpu,
                    flashAttn = flashAttn,
                    vaeDecodeOnly = vaeDecodeOnly,
                    flowShift = flowShift,
                    loraModelDir = loraModelDir,
                    loraApplyMode = loraApplyMode,
                )
            if (handle != 0L) {
                break
            }
            if (backend != ComputeBackend.CPU) {
                AndroidLogAdapter.w(LOG_TAG, "nativeCreate failed on $backend; retrying with CPU backend")
            }
        }
        if (handle == 0L && !effectiveOffloadToCpu) {
            AndroidLogAdapter.w(LOG_TAG, "nativeCreate failed on CPU backend; retrying with CPU offload")
            effectiveOffloadToCpu = true
            effectiveKeepClipOnCpu = true
            effectiveKeepVaeOnCpu = true
            handle =
                nativeCreateOrThrow(
                    modelPath = resolved.modelPath,
                    vaePath = resolved.vaePath,
                    t5xxlPath = resolved.t5xxlPath,
                    taesdPath = taesdPath,
                    nThreads = nThreads,
                    enableOpenCl = false,
                    useVulkan = false,
                    offloadToCpu = effectiveOffloadToCpu,
                    keepClipOnCpu = effectiveKeepClipOnCpu,
                    keepVaeOnCpu = effectiveKeepVaeOnCpu,
                    flashAttn = flashAttn,
                    vaeDecodeOnly = vaeDecodeOnly,
                    flowShift = flowShift,
                    loraModelDir = loraModelDir,
                    loraApplyMode = loraApplyMode,
                )
        }
        return handle
    }

    private fun nativeCreateOrThrow(
        modelPath: String,
        vaePath: String?,
        t5xxlPath: String?,
        taesdPath: String?,
        nThreads: Int,
        enableOpenCl: Boolean,
        useVulkan: Boolean,
        offloadToCpu: Boolean,
        keepClipOnCpu: Boolean,
        keepVaeOnCpu: Boolean,
        flashAttn: Boolean,
        vaeDecodeOnly: Boolean,
        flowShift: Float,
        loraModelDir: String?,
        loraApplyMode: LoraApplyMode,
    ): Long =
        try {
            StableDiffusion.supportNativeCreate(
                modelPath,
                vaePath,
                t5xxlPath,
                taesdPath,
                nThreads,
                enableOpenCl,
                useVulkan,
                offloadToCpu,
                keepClipOnCpu,
                keepVaeOnCpu,
                flashAttn,
                vaeDecodeOnly,
                flowShift,
                loraModelDir,
                loraApplyMode.id,
            )
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
