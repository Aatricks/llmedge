package io.aatricks.llmedge.model

import android.content.Context
import io.aatricks.llmedge.core.LLMEdgeException
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.huggingface.HuggingFaceHub
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
        spec.hints.conversion?.let { return resolveConvertedModel(context, spec, it) }
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

private fun resolveConvertedModel(
    context: Context,
    spec: ModelSpec,
    conversion: ModelConversion,
): File {
    val target = convertedModelTarget(context, spec, conversion)
    if (target.isFile && target.length() > 0L) {
        return ModelFileValidator.requireReadableFile(target, "Converted model")
    }
    throw LLMEdgeException(convertedModelInstructions(spec, conversion, target))
}

private fun convertedModelInstructions(
    spec: ModelSpec,
    conversion: ModelConversion,
    target: File,
): String {
    val adapterFlag = conversion.adapter.cliFlag?.let { " --adapter $it" }.orEmpty()
    return buildString {
        append("Safetensors model '${convertedSourceLabel(spec)}' has no converted GGUF yet, ")
        append("and on-device conversion is not available in this build.\n")
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
