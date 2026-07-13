package io.aatricks.llmedge.huggingface

internal object HFFileSelectionSupport {
    const val DEFAULT_MODELS_DIRECTORY = "hf-models"

    val DEFAULT_QUANTIZATION_PRIORITIES: List<String> =
        listOf(
            "Q4_K_M",
            "Q4_K",
            "Q4_K_S",
            "Q4_0",
            "Q3_K_L",
            "Q5_K_S",
            "Q3_K_M",
            "Q5_K_M",
            "Q3_K_S",
            "Q2_K",
            "Q5_K",
            "Q5_0",
            "Q8_0",
            ".gguf",
        )

    fun sanitize(modelId: String): String = modelId.replace("/", "_")

    fun selectModelFile(
        files: List<HFModelTree.HFModelFile>,
        filename: String?,
        preferredQuantizations: List<String>,
    ): HFModelTree.HFModelFile? {
        val allFiles = files.filter { it.type == "file" || it.type == null }
        if (!filename.isNullOrEmpty()) {
            allFiles.firstOrNull { it.path.equals(filename, ignoreCase = true) }?.let { return it }
            allFiles.firstOrNull { it.path.endsWith(filename, ignoreCase = true) }?.let { return it }
            allFiles.firstOrNull {
                it.path.substringAfterLast('/').equals(filename, ignoreCase = true)
            }?.let { return it }
        }

        val ggufCandidates = allFiles.filter { it.path.endsWith(".gguf", ignoreCase = true) }
        if (ggufCandidates.isEmpty()) {
            return null
        }

        preferredQuantizations.forEach { marker ->
            ggufCandidates.firstOrNull { it.path.contains(marker, ignoreCase = true) }?.let {
                return it
            }
        }

        return ggufCandidates.minByOrNull { it.size ?: it.lfs?.size ?: Long.MAX_VALUE }
    }

    fun selectRepoFile(
        files: List<HFModelTree.HFModelFile>,
        filename: String?,
        allowedExtensions: List<String>,
    ): HFModelTree.HFModelFile? {
        val allFiles = files.filter { it.type == "file" || it.type == null }
        if (!filename.isNullOrEmpty()) {
            allFiles.firstOrNull {
                it.path.equals(filename, ignoreCase = true) ||
                    it.path.endsWith(filename, ignoreCase = true)
            }?.let { return it }
            // An explicit filename that is not in the repo must fail the resolve; falling
            // through to the size heuristic silently downloads an unrelated checkpoint.
            return null
        }

        return allFiles
            .filter { candidate ->
                allowedExtensions.any { ext -> candidate.path.endsWith(ext, ignoreCase = true) }
            }
            .maxByOrNull { it.lfs?.size ?: it.size ?: 0L }
    }

    fun toMetadata(modelFile: HFModelTree.HFModelFile): HuggingFaceHub.ModelFileMetadata =
        HuggingFaceHub.ModelFileMetadata(
            path = modelFile.path,
            sizeBytes = modelFile.lfs?.size ?: modelFile.size ?: 0L,
            sha256 = modelFile.lfs?.oid ?: modelFile.oid,
        )
}
