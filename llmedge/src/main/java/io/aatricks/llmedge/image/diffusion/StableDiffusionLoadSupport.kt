package io.aatricks.llmedge.image.diffusion

import android.content.Context
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.huggingface.WanModelEntry
import io.aatricks.llmedge.huggingface.WanModelRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal data class StableDiffusionResolvedAssets(
    val modelPath: String,
    val vaePath: String?,
    val t5xxlPath: String?,
    val metadata: VideoModelMetadata,
)

internal object StableDiffusionLoadSupport {
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
