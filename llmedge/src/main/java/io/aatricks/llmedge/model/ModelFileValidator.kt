package io.aatricks.llmedge.model

import android.content.Context
import io.aatricks.llmedge.core.InvalidModelFileException
import io.aatricks.llmedge.core.ModelFileNotFoundException
import java.io.File
import java.io.FileInputStream

object ModelFileValidator {
    private val ggufMagic =
        byteArrayOf('G'.code.toByte(), 'G'.code.toByte(), 'U'.code.toByte(), 'F'.code.toByte())

    @JvmStatic
    fun requireReadableFile(path: String, modelKind: String = "Model"): File =
        requireReadableFile(File(path), modelKind)

    @JvmStatic
    fun requireReadableFile(file: File, modelKind: String = "Model"): File {
        if (!file.exists()) {
            throw ModelFileNotFoundException(file.absolutePath, modelKind)
        }
        if (!file.isFile) {
            throw InvalidModelFileException(file.absolutePath, "$modelKind path must point to a file.")
        }
        if (!file.canRead()) {
            throw InvalidModelFileException(file.absolutePath, "$modelKind file is not readable.")
        }
        if (file.length() <= 0L) {
            throw InvalidModelFileException(file.absolutePath, "$modelKind file is empty.")
        }
        return file
    }

    @JvmStatic
    fun requireReadableDirectory(path: String, modelKind: String): File {
        val directory = File(path)
        if (!directory.exists()) {
            throw ModelFileNotFoundException(directory.absolutePath, modelKind)
        }
        if (!directory.isDirectory) {
            throw InvalidModelFileException(
                directory.absolutePath,
                "$modelKind path must point to a directory.",
            )
        }
        if (!directory.canRead()) {
            throw InvalidModelFileException(
                directory.absolutePath,
                "$modelKind directory is not readable.",
            )
        }
        return directory
    }

    @JvmStatic
    fun requireGgufFile(path: String, modelKind: String = "GGUF model"): File {
        val file = requireReadableFile(path, modelKind)
        val header = ByteArray(ggufMagic.size)
        FileInputStream(file).use { input ->
            if (input.read(header) != ggufMagic.size || !header.contentEquals(ggufMagic)) {
                throw InvalidModelFileException(
                    file.absolutePath,
                    "$modelKind does not start with a valid GGUF header.",
                )
            }
        }
        return file
    }

    /**
     * Rejects an all-in-one checkpoint for a caller that loads the text encoders and VAE
     * separately. Such a file lands in `diffusion_model_path`, so its baked-in encoders and the
     * separately supplied ones are both loaded — a misconfiguration that surfaces later as a
     * generation-time crash rather than a load error, which makes it near-impossible to diagnose
     * from an app log alone.
     *
     * A file whose components cannot be determined passes: this exists to explain a known
     * mistake, not to gate models.
     */
    @JvmStatic
    fun requireDiffusionOnlyGguf(
        file: File,
        displayName: String = file.name,
    ): File {
        val summary = GgufFileSummary.read(file) ?: return file
        if (!summary.isAllInOne) return file
        val bundled =
            summary.components
                .filter { it != GgufComponent.DIFFUSION }
                .sorted()
                .joinToString(" and ") {
                    when (it) {
                        GgufComponent.TEXT_ENCODER -> "text encoders"
                        GgufComponent.VAE -> "a VAE"
                        GgufComponent.DIFFUSION -> "a denoiser"
                    }
                }
        throw InvalidModelFileException(
            file.absolutePath,
            "$displayName is an all-in-one checkpoint (it bundles $bundled). This preset " +
                "downloads those separately and needs a diffusion-model-only file. Pick a " +
                "DiT-only GGUF, or switch to a preset that takes all-in-one checkpoints.",
        )
    }

    @JvmStatic
    fun resolveReadableFile(
        context: Context,
        path: String,
        modelKind: String = "Model",
    ): File {
        val candidates =
            listOf(
                File(path),
                File(context.cacheDir, path),
                File(context.filesDir, path),
            )
        for (candidate in candidates) {
            if (candidate.exists()) {
                return requireReadableFile(candidate, modelKind)
            }
        }
        throw ModelFileNotFoundException(path, modelKind)
    }
}
