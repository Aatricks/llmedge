package io.aatricks.llmedge.huggingface

import android.content.Context
import android.util.Log
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal object HFDownloadSupport {
    private const val LOG_TAG = "HuggingFaceHub"

    data class ResolvedModel(
        val requestedModelId: String,
        val requestedRevision: String,
        val modelId: String,
        val revision: String,
        val aliasApplied: Boolean,
    )

    data class DownloadTarget(
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
        ): HuggingFaceHub.ModelDownloadResult =
            HuggingFaceHub.ModelDownloadResult(
                requestedModelId = requestedModelId,
                requestedRevision = requestedRevision,
                modelId = resolved.modelId,
                revision = resolved.revision,
                file = targetFile,
                fileInfo = HFFileSelectionSupport.toMetadata(modelFile),
                fromCache = fromCache,
                aliasApplied = resolved.aliasApplied,
            )
    }

    fun resolveModelReference(modelId: String, revision: String): ResolvedModel =
        ResolvedModel(
            requestedModelId = modelId,
            requestedRevision = revision,
            modelId = modelId,
            revision = revision,
            aliasApplied = false,
        )

    fun defaultModelsRoot(context: Context): File =
        File(context.filesDir, HFFileSelectionSupport.DEFAULT_MODELS_DIRECTORY)

    fun systemDownloadContext(context: Context, preferSystemDownloader: Boolean): Context? =
        if (preferSystemDownloader) context else null

    suspend fun ensureFileOnDisk(
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
    ): HuggingFaceHub.ModelDownloadResult = withContext(Dispatchers.IO) {
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

    fun isFileValidCached(targetFile: File, expectedSize: Long?, expectedSha: String?): Boolean {
        if (!targetFile.exists() || !targetFile.isFile) return false

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
                return false
            } catch (_: Throwable) {
                // Fall back to size-based checks when hashing is unavailable.
            }
        }

        if (expectedSize != null && expectedSize > 0L) {
            return targetFile.length() == expectedSize
        }

        return targetFile.length() > 0L
    }

    private fun buildDownloadTarget(
        destinationRoot: File,
        resolved: ResolvedModel,
        modelFile: HFModelTree.HFModelFile,
    ): DownloadTarget {
        val revisionDir = File(destinationRoot, "${HFFileSelectionSupport.sanitize(resolved.modelId)}/${resolved.revision}")
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
                    "${HFFileSelectionSupport.sanitize(target.modelFile.path)}-${System.currentTimeMillis()}.tmp",
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
                    throw IllegalStateException("Downloaded file sha mismatch for ${target.modelFile.path}")
                }
            } catch (_: Throwable) {
                // Size validation above is still a strong signal if hashing is unavailable.
            }
        }
    }

    private fun computeSha256(file: File): String {
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
    }
}
