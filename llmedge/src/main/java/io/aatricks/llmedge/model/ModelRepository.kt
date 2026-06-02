package io.aatricks.llmedge.model

import android.content.Context
import io.aatricks.llmedge.core.LLMEdgeException
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.text.runtime.SmolLM
import java.io.File

interface ModelRepository {
    /**
     * Resolve a [ModelSpec] to a readable local file.
     *
     * Preferred application path: `LLMEdge.create(...).models.resolve(...)` or
     * `edge.models.prefetch(...)`. Implement this interface when you need a custom acquisition
     * policy; call `HuggingFaceHub` directly only for advanced workflows that genuinely need
     * artifact-level control.
     */
    suspend fun resolve(
        context: Context,
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)? = null,
    ): File

    suspend fun prefetch(
        context: Context,
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)? = null,
    ): File = resolve(context, spec, onProgress)
}

class DefaultModelRepository(
    private val huggingFaceResolver: suspend (
        context: Context,
        spec: ModelSpec.HuggingFace,
        onProgress: ((ProgressEvent.Downloading) -> Unit)?,
    ) -> File = ::resolveHuggingFaceModel,
) : ModelRepository {
    override suspend fun resolve(
        context: Context,
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)?,
    ): File {
        spec.hints.conversion?.let { return resolveConvertedModel(context, spec, it, onProgress) }
        return when (spec) {
            is ModelSpec.LocalFile -> {
                ModelFileValidator.requireReadableFile(spec.file, "Model")
            }

            is ModelSpec.HuggingFace ->
                ModelFileValidator.requireReadableFile(huggingFaceResolver(context, spec, onProgress), "Model")
        }
    }
}

private const val CONVERTED_MODELS_DIR = "llmedge-converted"

private fun convertedSourceLabel(spec: ModelSpec): String =
    when (spec) {
        is ModelSpec.HuggingFace -> "${spec.repoId}@${spec.revision}"
        is ModelSpec.LocalFile -> spec.file.absolutePath
    }

private fun convertedSourceArg(spec: ModelSpec): String =
    when (spec) {
        is ModelSpec.HuggingFace -> spec.repoId
        is ModelSpec.LocalFile -> spec.file.absolutePath
    }

internal fun convertedModelTarget(
    context: Context,
    spec: ModelSpec,
    conversion: ModelConversion,
): File {
    val sanitized = convertedSourceLabel(spec).replace(Regex("[^A-Za-z0-9._-]"), "_")
    return File(File(context.filesDir, CONVERTED_MODELS_DIR), "${sanitized}__${conversion.precision.ggufLabel}.gguf")
}

// Files the on-device converter needs co-located in the model dir. Required ones must download;
// optional ones are best-effort (some tokenizers omit special_tokens_map.json).
private val CONVERTER_REQUIRED_FILES = listOf("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
private val CONVERTER_OPTIONAL_FILES = listOf("special_tokens_map.json")
private val CONVERTER_FILE_EXTENSIONS = listOf(".json", ".safetensors", ".txt", ".model")

private suspend fun resolveConvertedModel(
    context: Context,
    spec: ModelSpec,
    conversion: ModelConversion,
    onProgress: ((ProgressEvent.Downloading) -> Unit)?,
): File {
    val target = convertedModelTarget(context, spec, conversion)
    if (target.isFile && target.length() > 0L) {
        return ModelFileValidator.requireReadableFile(target, "Converted model")
    }

    // A text GGUF without a baked tokenizer is not loadable, so the pre-tokenizer id is mandatory.
    // Fail before downloading anything when it is absent.
    val tokenizerPre = conversion.tokenizerPre
    if (tokenizerPre.isNullOrBlank()) {
        throw LLMEdgeException(
            "On-device conversion of '${convertedSourceLabel(spec)}' needs ModelConversion.tokenizerPre " +
                "(the tokenizer.ggml.pre id, e.g. \"smollm\"), which was not provided.\n" +
                convertedModelInstructions(spec, conversion, target),
        )
    }

    val modelDir = prepareSafetensorsDir(context, spec, onProgress)
    if (!File(modelDir, "model.safetensors").isFile) {
        throw LLMEdgeException(
            "Source '${convertedSourceLabel(spec)}' has no single-file model.safetensors in " +
                "${modelDir.absolutePath} (sharded or non-safetensors layouts are not supported by the " +
                "on-device converter).\n" + convertedModelInstructions(spec, conversion, target),
        )
    }

    target.parentFile?.mkdirs()
    val tmp = File(target.parentFile, "${target.name}.tmp${System.nanoTime()}")
    try {
        SmolLM.convertSafetensorsToGguf(modelDir.absolutePath, tmp.absolutePath, tokenizerPre)
    } catch (_: UnsatisfiedLinkError) {
        tmp.delete()
        throw LLMEdgeException(convertedModelInstructions(spec, conversion, target))
    } catch (t: Throwable) {
        tmp.delete()
        throw t
    }
    if (!tmp.isFile || tmp.length() == 0L) {
        tmp.delete()
        throw LLMEdgeException("On-device conversion produced no output for '${convertedSourceLabel(spec)}'.")
    }
    // Publish atomically: a crash mid-convert leaves only the temp file, never a corrupt cached target.
    if (!tmp.renameTo(target)) {
        tmp.copyTo(target, overwrite = true)
        tmp.delete()
    }
    return ModelFileValidator.requireReadableFile(target, "Converted model")
}

/** Provide a local directory containing config.json + model.safetensors + tokenizer files. */
private suspend fun prepareSafetensorsDir(
    context: Context,
    spec: ModelSpec,
    onProgress: ((ProgressEvent.Downloading) -> Unit)?,
): File =
    when (spec) {
        is ModelSpec.LocalFile -> {
            require(spec.file.isDirectory) {
                "safetensorsLocal expects a directory with config.json + model.safetensors, got: ${spec.file.absolutePath}"
            }
            spec.file
        }

        is ModelSpec.HuggingFace -> downloadSafetensorsDir(context, spec, onProgress)
    }

/** Download the converter's input files for [spec] into one dir and return it (the flat HF cache dir). */
private suspend fun downloadSafetensorsDir(
    context: Context,
    spec: ModelSpec.HuggingFace,
    onProgress: ((ProgressEvent.Downloading) -> Unit)?,
): File {
    var modelDir: File? = null
    for (name in CONVERTER_REQUIRED_FILES) {
        val result =
            HuggingFaceHub.ensureRepoFileOnDisk(
                context = context,
                modelId = spec.repoId,
                revision = spec.revision,
                filename = name,
                allowedExtensions = CONVERTER_FILE_EXTENSIONS,
                token = spec.token,
                forceDownload = spec.forceDownload,
                preferSystemDownloader = spec.preferSystemDownloader,
                onProgress = onProgress?.asByteProgress(spec),
            )
        if (modelDir == null) modelDir = result.file.parentFile
    }
    for (name in CONVERTER_OPTIONAL_FILES) {
        runCatching {
            HuggingFaceHub.ensureRepoFileOnDisk(
                context = context,
                modelId = spec.repoId,
                revision = spec.revision,
                filename = name,
                allowedExtensions = CONVERTER_FILE_EXTENSIONS,
                token = spec.token,
                forceDownload = spec.forceDownload,
                preferSystemDownloader = spec.preferSystemDownloader,
                onProgress = onProgress?.asByteProgress(spec),
            )
        }
    }
    return requireNotNull(modelDir) { "No converter input files downloaded for ${spec.repoId}." }
}

private fun convertedModelInstructions(
    spec: ModelSpec,
    conversion: ModelConversion,
    target: File,
): String {
    val adapterFlag = conversion.adapter.cliFlag?.let { " --adapter $it" }.orEmpty()
    return buildString {
        append("Safetensors model '${convertedSourceLabel(spec)}' has no converted GGUF, and on-device ")
        append("conversion did not run (unavailable in this build, or unsupported model).\n")
        append("Convert it offline on a dev box/CI:\n")
        append("  python tools/safetensors-convert/convert.py --source ${convertedSourceArg(spec)}")
        append("$adapterFlag --precision ${conversion.precision.ggufLabel} --out model.gguf\n")
        append("then place the result at:\n  ${target.absolutePath}\n")
        append("(or load a pre-converted GGUF directly with ModelSpec.localFile(...)).")
    }
}

class BoundModelRepository internal constructor(
    private val context: Context,
    private val repository: ModelRepository,
) {
    /** Facade-scoped model access path used by `LLMEdge.create(...).models`. */
    suspend fun resolve(
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)? = null,
    ): File = repository.resolve(context, spec, onProgress)

    suspend fun prefetch(
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)? = null,
    ): File = repository.prefetch(context, spec, onProgress)
}

private fun ModelSpec.HuggingFace.shouldResolveAsRepoFile(): Boolean {
    when (hints.artifactKind) {
        ModelArtifactKind.REPO_FILE,
        ModelArtifactKind.PROJECTOR,
        ModelArtifactKind.DIFFUSION_MODEL,
        ModelArtifactKind.VAE,
        ModelArtifactKind.TEXT_ENCODER,
        ModelArtifactKind.TAEHV,
            -> return true

        ModelArtifactKind.GGUF_MODEL -> return false
        ModelArtifactKind.AUTO -> Unit
    }
    val explicitFilename = filename ?: return false
    return !explicitFilename.endsWith(".gguf", ignoreCase = true)
}

private suspend fun resolveHuggingFaceModel(
    context: Context,
    spec: ModelSpec.HuggingFace,
    onProgress: ((ProgressEvent.Downloading) -> Unit)?,
): File {
    return if (spec.shouldResolveAsRepoFile()) {
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
            onProgress = onProgress?.asByteProgress(spec),
        ).file
    } else {
        HuggingFaceHub.ensureModelOnDisk(
            context = context,
            modelId = spec.repoId,
            revision = spec.revision,
            preferredQuantizations = spec.preferredQuantizations,
            filename = spec.filename,
            token = spec.token,
            forceDownload = spec.forceDownload,
            preferSystemDownloader = spec.preferSystemDownloader,
            onProgress = onProgress?.asByteProgress(spec),
        ).file
    }
}

private fun ((ProgressEvent.Downloading) -> Unit).asByteProgress(
    spec: ModelSpec.HuggingFace,
): (downloaded: Long, total: Long?) -> Unit =
    { downloaded, total ->
        invoke(
            ProgressEvent.Downloading(
                model = spec,
                downloadedBytes = downloaded,
                totalBytes = total,
            ),
        )
    }
