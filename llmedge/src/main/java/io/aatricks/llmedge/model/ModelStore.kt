package io.aatricks.llmedge.model

import android.content.Context
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File

interface ModelStore {
    suspend fun resolve(
        context: Context,
        spec: ModelSpec.HuggingFace,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): File
}

class HuggingFaceModelStore : ModelStore {
    override suspend fun resolve(
        context: Context,
        spec: ModelSpec.HuggingFace,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)?,
    ): File {
        return try {
            HuggingFaceHub.ensureModelOnDisk(
                context = context,
                modelId = spec.repoId,
                revision = spec.revision,
                preferredQuantizations = spec.preferredQuantizations,
                filename = spec.filename,
                token = spec.token,
                forceDownload = spec.forceDownload,
                preferSystemDownloader = spec.preferSystemDownloader,
                onProgress = onProgress,
            ).file
        } catch (_: IllegalArgumentException) {
            val filename =
                requireNotNull(spec.filename) {
                    "A filename is required when resolving ${spec.repoId} through ensureRepoFileOnDisk()."
                }
            HuggingFaceHub.ensureRepoFileOnDisk(
                context = context,
                modelId = spec.repoId,
                revision = spec.revision,
                filename = filename,
                allowedExtensions = listOf(".gguf", ".bin", ".safetensors", ".ckpt", ".pt"),
                token = spec.token,
                forceDownload = spec.forceDownload,
                preferSystemDownloader = spec.preferSystemDownloader,
                onProgress = onProgress,
            ).file
        }
    }
}
