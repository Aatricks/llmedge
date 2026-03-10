/*
 * Copyright (C) 2025 Aatricks
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.aatricks.llmedge.huggingface

import android.content.Context
import android.util.Log
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** High-level helper to discover and download GGUF models from Hugging Face. */
object HuggingFaceHub {
    data class ModelDownloadResult(
        val requestedModelId: String,
        val requestedRevision: String,
        val modelId: String,
        val revision: String,
        val file: File,
        val fileInfo: ModelFileMetadata,
        val fromCache: Boolean,
        val aliasApplied: Boolean,
    )

    data class ModelFileMetadata(
        val path: String,
        val sizeBytes: Long,
        val sha256: String?,
    )

    suspend fun ensureModelOnDisk(
        context: Context,
        modelId: String,
        revision: String = "main",
        preferredQuantizations: List<String> = DEFAULT_QUANTIZATION_PRIORITIES,
        filename: String? = null,
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = false,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): ModelDownloadResult =
        ensureModelOnDisk(
            destinationRoot = defaultModelsRoot(context),
            modelId = modelId,
            revision = revision,
            preferredQuantizations = preferredQuantizations,
            filename = filename,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            systemDownloadContext = systemDownloadContext(context, preferSystemDownloader),
            onProgress = onProgress,
        )

    suspend fun ensureModelOnDisk(
        destinationRoot: File,
        modelId: String,
        revision: String = "main",
        preferredQuantizations: List<String> = DEFAULT_QUANTIZATION_PRIORITIES,
        filename: String? = null,
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = false,
        systemDownloadContext: Context? = null,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): ModelDownloadResult {
        return ensureFileOnDisk(
            destinationRoot = destinationRoot,
            modelId = modelId,
            revision = revision,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            systemDownloadContext = systemDownloadContext,
            onProgress = onProgress,
            noMatchMessage = "No GGUF file found for '$modelId' (revision '$revision')",
            fileSelector = { files -> selectModelFile(files, filename, preferredQuantizations) },
        )
    }

    suspend fun ensureWanAssetsOnDisk(
        context: Context,
        wanModelId: String,
        preferSystemDownloader: Boolean = true,
        token: String? = null,
        forceDownload: Boolean = false,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): Triple<ModelDownloadResult, ModelDownloadResult?, ModelDownloadResult?> =
        withContext(Dispatchers.IO) {
            var registryEntry = WanModelRegistry.findById(context, wanModelId)
            if (registryEntry == null) {
                // Try prefix match (e.g. 'wan/Wan2.1-T2V-1.3B' vs 'Wan2.1-T2V-1.3B')
                val trimmed = wanModelId.removePrefix("wan/")
                registryEntry = WanModelRegistry.findByModelIdPrefix(context, trimmed)
            }
            registryEntry ?: throw IllegalArgumentException("Unknown Wan model $wanModelId")

            val modelRes =
                ensureModelOnDisk(
                    context = context,
                    modelId = registryEntry.modelId,
                    filename = registryEntry.filename,
                    token = token,
                    forceDownload = forceDownload,
                    preferSystemDownloader = preferSystemDownloader,
                    onProgress = onProgress,
                )

            val vaeRes =
                registryEntry.vaeFilename?.let { vaeName ->
                    ensureRepoFileOnDisk(
                        context = context,
                        modelId = registryEntry.modelId,
                        filename = vaeName,
                        allowedExtensions = listOf(".safetensors", ".pt"),
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = preferSystemDownloader,
                        onProgress = onProgress,
                    )
                }

            val t5Res =
                registryEntry.t5ModelId?.let { t5ModelId ->
                    val t5Filename =
                        registryEntry.t5Filename
                            ?: throw IllegalArgumentException(
                                "Registry entry for $wanModelId missing t5 filename"
                            )
                    ensureRepoFileOnDisk(
                        context = context,
                        modelId = t5ModelId,
                        filename = t5Filename,
                        allowedExtensions = listOf(".gguf"),
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = preferSystemDownloader,
                        onProgress = onProgress,
                    )
                }

            Triple(modelRes, vaeRes, t5Res)
        }

    /* Cache utilities */
    fun clearCache(context: Context) {
        val root = File(context.filesDir, DEFAULT_MODELS_DIRECTORY)
        if (root.exists()) {
            root.deleteRecursively()
        }
    }

    fun listCachedModels(context: Context): List<File> {
        val root = File(context.filesDir, DEFAULT_MODELS_DIRECTORY)
        return if (root.exists() && root.isDirectory) {
            root.listFiles()?.filter { it.isDirectory }?.toList() ?: emptyList()
        } else emptyList()
    }

    /**
     * Ensure an arbitrary file from a Hugging Face model repo is present on disk. This is useful
     * for files that are not GGUF models (for example VAE safetensors or other checkpoints) where
     * we want to specify an explicit filename or fall back to a heuristic (largest file with
     * allowed extensions).
     */
    suspend fun ensureRepoFileOnDisk(
        context: Context,
        modelId: String,
        revision: String = "main",
        filename: String? = null,
        allowedExtensions: List<String> =
            listOf(".safetensors", ".pt", ".ckpt", ".gguf", ".bin"),
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = false,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): ModelDownloadResult =
        ensureRepoFileOnDisk(
            destinationRoot = defaultModelsRoot(context),
            modelId = modelId,
            revision = revision,
            filename = filename,
            allowedExtensions = allowedExtensions,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            systemDownloadContext = systemDownloadContext(context, preferSystemDownloader),
            onProgress = onProgress,
        )

    suspend fun ensureRepoFileOnDisk(
        destinationRoot: File,
        modelId: String,
        revision: String = "main",
        filename: String? = null,
        allowedExtensions: List<String> =
            listOf(".safetensors", ".pt", ".ckpt", ".gguf", ".bin"),
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = false,
        systemDownloadContext: Context? = null,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): ModelDownloadResult {
        return ensureFileOnDisk(
            destinationRoot = destinationRoot,
            modelId = modelId,
            revision = revision,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            systemDownloadContext = systemDownloadContext,
            onProgress = onProgress,
            noMatchMessage = "No file found for '$modelId' matching ${filename ?: allowedExtensions}",
            fileSelector = { files -> selectRepoFile(files, filename, allowedExtensions) },
        )
    }

    fun sanitize(modelId: String): String = modelId.replace("/", "_")

    private fun resolveModelReference(modelId: String, revision: String): ResolvedModel {
        return ResolvedModel(
            requestedModelId = modelId,
            requestedRevision = revision,
            modelId = modelId,
            revision = revision,
            aliasApplied = false,
        )
    }

    private fun selectModelFile(
        files: List<HFModelTree.HFModelFile>,
        filename: String?,
        preferredQuantizations: List<String>,
    ): HFModelTree.HFModelFile? {
        // NOTE: The model specs endpoint (siblings list) does not populate a 'type' field.
        // Treat null type as a file entry.
        val allFiles = files.filter { it.type == "file" || it.type == null }
        
        // If a specific filename is provided, search all files (not just .gguf)
        if (!filename.isNullOrEmpty()) {
            allFiles.firstOrNull { it.path.equals(filename, ignoreCase = true) }?.let {
                return it
            }
            allFiles.firstOrNull { it.path.endsWith(filename, ignoreCase = true) }?.let {
                return it
            }
            // Also try matching just the filename part
            allFiles.firstOrNull { 
                it.path.substringAfterLast('/').equals(filename, ignoreCase = true) 
            }?.let {
                return it
            }
        }
        
        // For automatic selection, prefer GGUF files
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

    private fun selectRepoFile(
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
        }

        return allFiles
            .filter { candidate ->
                allowedExtensions.any { ext -> candidate.path.endsWith(ext, ignoreCase = true) }
            }
            .maxByOrNull { it.lfs?.size ?: it.size ?: 0L }
    }

    private const val DEFAULT_MODELS_DIRECTORY = "hf-models"
    private const val LOG_TAG = "HuggingFaceHub"

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

    private fun HFModelTree.HFModelFile.toMetadata(): ModelFileMetadata =
        ModelFileMetadata(
            path = path,
            sizeBytes = lfs?.size ?: size ?: 0L,
            sha256 = lfs?.oid ?: oid,
        )

    private suspend fun ensureFileOnDisk(
        destinationRoot: File,
        modelId: String,
        revision: String,
        token: String?,
        forceDownload: Boolean,
        preferSystemDownloader: Boolean,
        systemDownloadContext: Context?,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)?,
        noMatchMessage: String,
        fileSelector: (List<HFModelTree.HFModelFile>) -> HFModelTree.HFModelFile?,
    ): ModelDownloadResult = withContext(Dispatchers.IO) {
        val resolved = resolveModelReference(modelId, revision)
        val files = HFModels.tree().getModelFileTree(resolved.modelId, resolved.revision, token)
        val modelFile = fileSelector(files) ?: throw IllegalArgumentException(noMatchMessage)
        val target = buildDownloadTarget(destinationRoot, resolved, modelFile)

        if (!forceDownload && isFileValidCached(target.targetFile, target.expectedSize, target.expectedSha)) {
            Log.d(
                LOG_TAG,
                "Using cached file for ${resolved.modelId}@${resolved.revision}: ${target.targetFile.absolutePath}",
            )
            return@withContext target.toResult(modelId, revision, resolved, fromCache = true)
        }

        target.targetFile.parentFile?.mkdirs()
        maybeDownloadWithSystem(
            target = target,
            token = token,
            preferSystemDownloader = preferSystemDownloader,
            systemDownloadContext = systemDownloadContext,
            onProgress = onProgress,
        )

        if (!target.targetFile.exists()) {
            HFModels.download().downloadModelFile(
                modelId = resolved.modelId,
                revision = resolved.revision,
                filePath = modelFile.path,
                destination = target.targetFile,
                token = token,
                onProgress = onProgress,
            )
        }

        verifyDownloadedFile(target)
        target.toResult(modelId, revision, resolved, fromCache = false)
    }

    private fun buildDownloadTarget(
        destinationRoot: File,
        resolved: ResolvedModel,
        modelFile: HFModelTree.HFModelFile,
    ): DownloadTarget {
        val revisionDir = File(destinationRoot, "${sanitize(resolved.modelId)}/${resolved.revision}")
        val targetName = modelFile.path.substringAfterLast('/')
        return DownloadTarget(
            modelFile = modelFile,
            targetFile = File(revisionDir, targetName),
            expectedSize = modelFile.lfs?.size ?: modelFile.size,
            expectedSha = modelFile.lfs?.oid ?: modelFile.oid,
            downloadUrl = HFEndpoints.fileDownloadEndpoint(
                resolved.modelId,
                resolved.revision,
                modelFile.path,
            ),
        )
    }

    private suspend fun maybeDownloadWithSystem(
        target: DownloadTarget,
        token: String?,
        preferSystemDownloader: Boolean,
        systemDownloadContext: Context?,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)?,
    ) {
        if (!preferSystemDownloader || systemDownloadContext == null) {
            return
        }

        try {
            val tempDir = systemDownloadContext.getExternalFilesDir("hf-downloads")
            if (tempDir == null) {
                Log.w(
                    LOG_TAG,
                    "External downloads directory unavailable; falling back to in-app streaming",
                )
                return
            }

            val tempFile =
                File(
                    tempDir,
                    "${sanitize(target.modelFile.path)}-${System.currentTimeMillis()}.tmp",
                )
            val downloaded =
                SystemDownload.download(
                    context = systemDownloadContext,
                    url = target.downloadUrl,
                    token = token,
                    destination = tempFile,
                    displayName = target.targetFile.name,
                    onProgress = onProgress,
                )
            if (target.targetFile.exists()) {
                target.targetFile.delete()
            }
            downloaded.copyTo(target.targetFile, overwrite = true)
            downloaded.delete()
        } catch (t: Throwable) {
            Log.w(
                LOG_TAG,
                "System download failed (${t.message}) - falling back to in-app downloader",
            )
        }
    }

    private fun verifyDownloadedFile(target: DownloadTarget) {
        val expectedSize = target.expectedSize
        if ((expectedSize ?: -1L) > 0L && target.targetFile.length() != expectedSize) {
            target.targetFile.delete()
            throw IllegalStateException("Downloaded file size mismatch for ${target.modelFile.path}")
        }

        target.modelFile.lfs?.oid?.let { expectedShaValue ->
            try {
                val actualSha = computeSha256(target.targetFile)
                if (!actualSha.equals(expectedShaValue, ignoreCase = true)) {
                    target.targetFile.delete()
                    throw IllegalStateException(
                        "Downloaded file sha mismatch for ${target.modelFile.path}",
                    )
                }
            } catch (_: Throwable) {
                // Size validation above is still a strong signal if hashing is unavailable.
            }
        }
    }

    private fun defaultModelsRoot(context: Context): File =
        File(context.filesDir, DEFAULT_MODELS_DIRECTORY)

    private fun systemDownloadContext(context: Context, preferSystemDownloader: Boolean): Context? =
        if (preferSystemDownloader) context else null

    private data class ResolvedModel(
        val requestedModelId: String,
        val requestedRevision: String,
        val modelId: String,
        val revision: String,
        val aliasApplied: Boolean,
    )

    private data class DownloadTarget(
        val modelFile: HFModelTree.HFModelFile,
        val targetFile: File,
        val expectedSize: Long?,
        val expectedSha: String?,
        val downloadUrl: String,
    ) {
        fun toResult(
            requestedModelId: String,
            requestedRevision: String,
            resolved: ResolvedModel,
            fromCache: Boolean,
        ): ModelDownloadResult =
            ModelDownloadResult(
                requestedModelId = requestedModelId,
                requestedRevision = requestedRevision,
                modelId = resolved.modelId,
                revision = resolved.revision,
                file = targetFile,
                fileInfo = modelFile.toMetadata(),
                fromCache = fromCache,
                aliasApplied = resolved.aliasApplied,
            )
    }

    private fun computeSha256(file: File): String {
        try {
            val md = java.security.MessageDigest.getInstance("SHA-256")
            file.inputStream().use { fis ->
                val buffer = ByteArray(8 * 1024)
                var bytesRead = fis.read(buffer)
                while (bytesRead >= 0) {
                    md.update(buffer, 0, bytesRead)
                    bytesRead = fis.read(buffer)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        } catch (t: Throwable) {
            throw t
        }
    }

    // Internal helper for tests: determine whether an existing file satisfies size or sha constraints
    internal fun isFileValidCached(targetFile: File, expectedSize: Long?, expectedSha: String?): Boolean {
        if (!targetFile.exists() || !targetFile.isFile) return false

        // Prefer SHA-based validation when available; it's the strongest indication of file integrity.
        // Normalize common SHA prefixes and whitespace before comparing to the computed SHA.
        if (!expectedSha.isNullOrEmpty()) {
            val normalizedExpectedSha = expectedSha.trim()
                .removePrefix("sha256:")
                .removePrefix("SHA256:")
                .removePrefix("SHA-256:")
                .removePrefix("urn:sha1:")
                .trim()
                .lowercase()

            try {
                val actualSha = computeSha256(targetFile)
                if (actualSha.equals(normalizedExpectedSha, ignoreCase = true)) return true
                // SHA present but mismatch: consider the cache invalid
                return false
            } catch (_: Throwable) {
                // If hashing fails for any reason, fall back to alternative checks below
            }
        }

        if (expectedSize != null && expectedSize > 0L) {
            return targetFile.length() == expectedSize
        }

        // Fallback: if file exists and has a positive length, treat as cached material
        return targetFile.length() > 0L
    }
}
