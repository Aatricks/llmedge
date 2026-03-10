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
