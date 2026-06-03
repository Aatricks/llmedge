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
            (
                listOf(
                    "file://${file.absolutePath}",
                    "artifact=${hints.artifactKind.name}",
                    "capabilities=${hints.capabilities.map(ModelCapability::name).sorted().joinToString(",")}",
                ) + listOfNotNull(hints.conversion?.cacheToken)
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
            (
                listOf(
                    "hf",
                    repoId,
                    revision,
                    filename.orEmpty(),
                    preferredQuantizations.joinToString(","),
                    forceDownload.toString(),
                    "artifact=${hints.artifactKind.name}",
                    "capabilities=${hints.capabilities.map(ModelCapability::name).sorted().joinToString(",")}",
                ) + listOfNotNull(hints.conversion?.cacheToken)
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

        /**
         * A safetensors model on Hugging Face to be converted to GGUF (at [precision]) before loading.
         *
         * Resolution first looks for a pre-converted GGUF in the app cache. If absent and
         * [tokenizerPre] is set, the model dir is downloaded and converted on-device (Track B / Phase
         * B2); otherwise it fails with instructions for `tools/safetensors-convert`. [tokenizerPre] is
         * the `tokenizer.ggml.pre` id to bake (e.g. "smollm") and is required for on-device conversion.
         * Use [adapter] = [ConversionAdapter.BONSAI_QLINEAR] for Bonsai / QLlama models.
         */
        @JvmStatic
        @JvmOverloads
        fun safetensors(
            repoId: String,
            precision: ConversionPrecision = ConversionPrecision.F16,
            adapter: ConversionAdapter = ConversionAdapter.NONE,
            revision: String = "main",
            token: String? = null,
            capabilities: Set<ModelCapability> = setOf(ModelCapability.TEXT),
            tokenizerPre: String? = null,
        ): ModelSpec =
            HuggingFace(
                repoId = repoId,
                revision = revision,
                token = token,
                hints =
                    ModelHints(
                        capabilities = capabilities,
                        conversion = ModelConversion(precision, adapter, tokenizerPre),
                    ),
            )

        /**
         * A local safetensors model directory to be converted to GGUF before loading.
         * See [safetensors] for conversion behavior; [tokenizerPre] is required for on-device conversion.
         */
        @JvmStatic
        @JvmOverloads
        fun safetensorsLocal(
            path: String,
            precision: ConversionPrecision = ConversionPrecision.F16,
            adapter: ConversionAdapter = ConversionAdapter.NONE,
            capabilities: Set<ModelCapability> = setOf(ModelCapability.TEXT),
            tokenizerPre: String? = null,
        ): ModelSpec =
            LocalFile(
                file = File(path),
                hints =
                    ModelHints(
                        capabilities = capabilities,
                        conversion = ModelConversion(precision, adapter, tokenizerPre),
                    ),
            )
    }
}
