package io.aatricks.llmedge.huggingface

import android.content.Context
import java.io.File

internal sealed interface ArtifactRequest {
    val modelId: String
    val revision: String
    val token: String?
    val forceDownload: Boolean
    val preferSystemDownloader: Boolean
    val onProgress: ((downloaded: Long, total: Long?) -> Unit)?

    data class Model(
        override val modelId: String,
        override val revision: String = "main",
        val preferredQuantizations: List<String> = HuggingFaceHub.DEFAULT_QUANTIZATION_PRIORITIES,
        val filename: String? = null,
        override val token: String? = null,
        override val forceDownload: Boolean = false,
        override val preferSystemDownloader: Boolean = false,
        override val onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ) : ArtifactRequest

    data class RepoFile(
        override val modelId: String,
        override val revision: String = "main",
        val filename: String? = null,
        val allowedExtensions: List<String> =
            listOf(".safetensors", ".pt", ".ckpt", ".gguf", ".bin"),
        override val token: String? = null,
        override val forceDownload: Boolean = false,
        override val preferSystemDownloader: Boolean = false,
        override val onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ) : ArtifactRequest
}

internal data class ArtifactDownloadContext(
    val destinationRoot: File,
    val systemDownloadContext: Context?,
)

internal object HFArtifactRequestSupport {
    fun artifactDownloadContext(
        context: Context,
        preferSystemDownloader: Boolean,
    ): ArtifactDownloadContext =
        ArtifactDownloadContext(
            destinationRoot = HFDownloadSupport.defaultModelsRoot(context),
            systemDownloadContext = HFDownloadSupport.systemDownloadContext(context, preferSystemDownloader),
        )

    fun modelArtifactRequest(
        modelId: String,
        revision: String,
        preferredQuantizations: List<String>,
        filename: String?,
        token: String?,
        forceDownload: Boolean,
        preferSystemDownloader: Boolean,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)?,
    ): ArtifactRequest.Model =
        ArtifactRequest.Model(
            modelId = modelId,
            revision = revision,
            preferredQuantizations = preferredQuantizations,
            filename = filename,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            onProgress = onProgress,
        )

    fun repoFileArtifactRequest(
        modelId: String,
        revision: String,
        filename: String?,
        allowedExtensions: List<String>,
        token: String?,
        forceDownload: Boolean,
        preferSystemDownloader: Boolean,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)?,
    ): ArtifactRequest.RepoFile =
        ArtifactRequest.RepoFile(
            modelId = modelId,
            revision = revision,
            filename = filename,
            allowedExtensions = allowedExtensions,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            onProgress = onProgress,
        )
}
