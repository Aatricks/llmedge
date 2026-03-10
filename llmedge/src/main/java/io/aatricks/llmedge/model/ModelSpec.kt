package io.aatricks.llmedge.model

import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File

sealed interface ModelSpec {
    val cacheKey: String

    data class LocalFile(val file: File) : ModelSpec {
        override val cacheKey: String = "file://${file.absolutePath}"
    }

    data class HuggingFace(
        val repoId: String,
        val filename: String? = null,
        val revision: String = "main",
        val preferredQuantizations: List<String> = HuggingFaceHub.DEFAULT_QUANTIZATION_PRIORITIES,
        val token: String? = null,
        val forceDownload: Boolean = false,
        val preferSystemDownloader: Boolean = true,
    ) : ModelSpec {
        override val cacheKey: String =
            listOf(
                "hf",
                repoId,
                revision,
                filename.orEmpty(),
                preferredQuantizations.joinToString(","),
                forceDownload.toString(),
            ).joinToString("|")
    }

    companion object {
        @JvmStatic
        fun localFile(path: String): ModelSpec = LocalFile(File(path))

        @JvmStatic
        fun localFile(file: File): ModelSpec = LocalFile(file)

        @JvmStatic
        @JvmOverloads
        fun huggingFace(
            repoId: String,
            filename: String? = null,
            revision: String = "main",
            preferredQuantizations: List<String> = HuggingFaceHub.DEFAULT_QUANTIZATION_PRIORITIES,
            token: String? = null,
            forceDownload: Boolean = false,
            preferSystemDownloader: Boolean = true,
        ): ModelSpec =
            HuggingFace(
                repoId = repoId,
                filename = filename,
                revision = revision,
                preferredQuantizations = preferredQuantizations,
                token = token,
                forceDownload = forceDownload,
                preferSystemDownloader = preferSystemDownloader,
            )
    }
}
