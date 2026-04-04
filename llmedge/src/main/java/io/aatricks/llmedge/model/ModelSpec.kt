package io.aatricks.llmedge.model

import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File

sealed interface ModelSpec {
    val cacheKey: String
    val hints: ModelHints

    data class LocalFile(
        val file: File,
        override val hints: ModelHints = ModelHints(),
    ) : ModelSpec {
        override val cacheKey: String =
            listOf(
                "file://${file.absolutePath}",
                "artifact=${hints.artifactKind.name}",
                "capabilities=${hints.capabilities.map(ModelCapability::name).sorted().joinToString(",")}",
            ).joinToString("|")
    }

    data class HuggingFace(
        val repoId: String,
        val filename: String? = null,
        val revision: String = "main",
        val preferredQuantizations: List<String> = HuggingFaceHub.DEFAULT_QUANTIZATION_PRIORITIES,
        val token: String? = null,
        val forceDownload: Boolean = false,
        val preferSystemDownloader: Boolean = true,
        override val hints: ModelHints = ModelHints(),
    ) : ModelSpec {
        override val cacheKey: String =
            listOf(
                "hf",
                repoId,
                revision,
                filename.orEmpty(),
                preferredQuantizations.joinToString(","),
                forceDownload.toString(),
                "artifact=${hints.artifactKind.name}",
                "capabilities=${hints.capabilities.map(ModelCapability::name).sorted().joinToString(",")}",
            ).joinToString("|")
    }

    companion object {
        @JvmStatic
        @JvmOverloads
        fun localFile(
            path: String,
            hints: ModelHints = ModelHints(),
        ): ModelSpec = LocalFile(File(path), hints)

        @JvmStatic
        @JvmOverloads
        fun localFile(
            file: File,
            hints: ModelHints = ModelHints(),
        ): ModelSpec = LocalFile(file, hints)

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
            hints: ModelHints = ModelHints(),
        ): ModelSpec =
            HuggingFace(
                repoId = repoId,
                filename = filename,
                revision = revision,
                preferredQuantizations = preferredQuantizations,
                token = token,
                forceDownload = forceDownload,
                preferSystemDownloader = preferSystemDownloader,
                hints = hints,
            )
    }
}
