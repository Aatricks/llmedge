package io.aatricks.llmedge.huggingface

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal object HFWanAssetSupport {
    suspend fun ensureWanAssetsOnDisk(
        context: Context,
        wanModelId: String,
        preferSystemDownloader: Boolean = true,
        token: String? = null,
        forceDownload: Boolean = false,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): Triple<HuggingFaceHub.ModelDownloadResult, HuggingFaceHub.ModelDownloadResult?, HuggingFaceHub.ModelDownloadResult?> =
        withContext(Dispatchers.IO) {
            var registryEntry = WanModelRegistry.findById(context, wanModelId)
            if (registryEntry == null) {
                val trimmed = wanModelId.removePrefix("wan/")
                registryEntry = WanModelRegistry.findByModelIdPrefix(context, trimmed)
            }
            registryEntry ?: throw IllegalArgumentException("Unknown Wan model $wanModelId")

            val modelRes =
                HuggingFaceHub.ensureModelOnDisk(
                    context = context,
                    modelId = registryEntry.modelId,
                    filename = registryEntry.filename,
                    token = token,
                    forceDownload = forceDownload,
                    preferSystemDownloader = preferSystemDownloader,
                    onProgress = onProgress,
                )

            val vaeRes =
                registryEntry.vaeFilename?.let { vaeName ->
                    HuggingFaceHub.ensureRepoFileOnDisk(
                        context = context,
                        modelId = registryEntry.modelId,
                        filename = vaeName,
                        allowedExtensions = listOf(".safetensors", ".pt"),
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = preferSystemDownloader,
                        onProgress = onProgress,
                    )
                }

            val t5Res =
                registryEntry.t5ModelId?.let { t5ModelId ->
                    val t5Filename =
                        registryEntry.t5Filename
                            ?: throw IllegalArgumentException(
                                "Registry entry for $wanModelId missing t5 filename"
                            )
                    HuggingFaceHub.ensureRepoFileOnDisk(
                        context = context,
                        modelId = t5ModelId,
                        filename = t5Filename,
                        allowedExtensions = listOf(".gguf"),
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = preferSystemDownloader,
                        onProgress = onProgress,
                    )
                }

            Triple(modelRes, vaeRes, t5Res)
        }
}
