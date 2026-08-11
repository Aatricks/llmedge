package io.aatricks.llmedge.huggingface

import android.content.Context
import android.util.Log
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal object HFDownloadSupport {
    private const val LOG_TAG = "HuggingFaceHub"

    // Captured at class load (~process start). System-downloader temp files (.tmp) are
    // deleted in a coroutine finally block on success/cancellation, but that never runs
    // if the process is killed mid-download (low-memory killer, worker crash), leaving
    // multi-GB orphans behind. Any .tmp older than this timestamp belongs to a previous
    // process and is safe to delete; a .tmp being written by the current process is newer.
    private val PROCESS_START_MILLIS = System.currentTimeMillis()

    /**
     * Deletes system-download temp files orphaned by a previous process instance. Files
     * modified by the current process (active downloads) are newer than [olderThanMillis]
     * and are left untouched, as are [protectedPaths] — partials still tracked by
     * DownloadManager that we intend to resume. Returns the number of bytes reclaimed.
     */
    internal fun cleanupOrphanedTempFiles(
        tempDir: File,
        olderThanMillis: Long = PROCESS_START_MILLIS,
        protectedPaths: Set<String> = emptySet(),
    ): Long {
        val stale = tempDir.listFiles { f ->
            f.isFile &&
                f.name.endsWith(".tmp") &&
                f.lastModified() < olderThanMillis &&
                f.absolutePath !in protectedPaths
        } ?: return 0L
        var freedBytes = 0L
        for (file in stale) {
            val size = file.length()
            if (file.delete()) {
                freedBytes += size
            }
        }
        return freedBytes
    }

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
        /** Where builds before the nested-layout fix cached this file, when that differs. */
        val legacyFlatFile: File? = null,
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
        recursive: Boolean = false,
        fileSelector: (List<HFModelTree.HFModelFile>) -> HFModelTree.HFModelFile?,
    ): HuggingFaceHub.ModelDownloadResult = withContext(Dispatchers.IO) {
        val resolved = resolveModelReference(modelId, revision)
        val files = HFModels.tree().getModelFileTree(resolved.modelId, resolved.revision, token, recursive)
        val modelFile = fileSelector(files) ?: throw IllegalArgumentException(noMatchMessage)
        val target = buildDownloadTarget(destinationRoot, resolved, modelFile)
        deleteFlatLayoutLeftover(target).takeIf { it > 0L }?.let {
            Log.i(LOG_TAG, "Removed flat-layout cache leftover for ${modelFile.path}, reclaimed $it bytes")
        }

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

    /**
     * Reclaims the copy an older build left at the flat, basename-only cache path. It cannot be
     * reused — nothing records which repo subdirectory it came from — and for a diffusion
     * transformer it is multiple GB of dead weight on the device. Returns the bytes reclaimed.
     */
    internal fun deleteFlatLayoutLeftover(target: DownloadTarget): Long {
        val legacy = target.legacyFlatFile ?: return 0L
        if (!legacy.isFile) return 0L
        val freed = legacy.length()
        if (!legacy.delete()) return 0L
        File(legacy.parent, "${legacy.name}.validated").delete()
        return freed
    }

    fun isFileValidCached(targetFile: File, expectedSize: Long?, expectedSha: String?): Boolean {
        if (!targetFile.exists() || !targetFile.isFile) return false

        val expectedShaNormalized = expectedSha?.trim()
            ?.removePrefix("sha256:")
            ?.removePrefix("SHA256:")
            ?.removePrefix("SHA-256:")
            ?.removePrefix("urn:sha1:")
            ?.trim()
            ?.lowercase()

        if (!expectedShaNormalized.isNullOrEmpty()) {
            val markerFile = File(targetFile.parent, "${targetFile.name}.validated")
            var markerIsValid = false
            if (markerFile.exists() && markerFile.isFile) {
                try {
                    val lines = markerFile.readLines()
                    if (lines.size >= 3) {
                        val cachedSha = lines[0].trim()
                        val cachedLength = lines[1].trim().toLong()
                        val cachedMtime = lines[2].trim().toLong()
                        if (cachedLength == targetFile.length() &&
                            cachedMtime == targetFile.lastModified() &&
                            cachedSha.equals(expectedShaNormalized, ignoreCase = true)) {
                            markerIsValid = true
                        }
                    }
                } catch (_: Throwable) {
                    // Fall back to re-hashing
                }
            }

            if (markerIsValid) {
                return true
            }

            try {
                val actualSha = computeSha256(targetFile)
                if (actualSha.equals(expectedShaNormalized, ignoreCase = true)) {
                    writeMarkerFile(targetFile, actualSha)
                    return true
                }
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

    private fun writeMarkerFile(file: File, sha: String) {
        try {
            val markerFile = File(file.parent, "${file.name}.validated")
            markerFile.writeText("$sha\n${file.length()}\n${file.lastModified()}")
        } catch (_: Throwable) {
            // ignore
        }
    }

    private fun buildDownloadTarget(
        destinationRoot: File,
        resolved: ResolvedModel,
        modelFile: HFModelTree.HFModelFile,
    ): DownloadTarget {
        val revisionDir = File(destinationRoot, "${HFFileSelectionSupport.sanitize(resolved.modelId)}/${resolved.revision}")
        // Mirror the repo's directory layout: repos like MiniT2I/MiniT2I ship the same basename
        // under several variant folders, and flattening to the basename makes them collide.
        val targetName = HFFileSelectionSupport.relativeCachePath(modelFile.path)
        val flatName = modelFile.path.substringAfterLast('/')
        return DownloadTarget(
            modelFile = modelFile,
            targetFile = File(revisionDir, targetName),
            legacyFlatFile = File(revisionDir, flatName).takeIf { targetName != flatName },
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

            // Reclaim multi-GB temp files leaked by a previous process that was killed
            // mid-download before its cleanup could run, but keep any partial that
            // DownloadManager is still tracking so an interrupted download can resume.
            val reclaimedBytes =
                cleanupOrphanedTempFiles(
                    tempDir,
                    protectedPaths = SystemDownload.resumableTempPaths(systemDownloadContext),
                )
            if (reclaimedBytes > 0L) {
                Log.i(LOG_TAG, "Removed orphaned download temp files, reclaimed $reclaimedBytes bytes")
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
            val publishTemp = File(target.targetFile.parentFile ?: target.targetFile.absoluteFile.parentFile, "${target.targetFile.name}.publish.tmp")
            val moved = downloaded.renameTo(publishTemp)
            if (!moved) {
                downloaded.copyTo(publishTemp, overwrite = true)
                downloaded.delete()
            }
            if (target.targetFile.exists()) {
                target.targetFile.delete()
            }
            if (!publishTemp.renameTo(target.targetFile)) {
                publishTemp.delete()
                throw IllegalStateException("Failed to publish system download to ${target.targetFile.name}")
            }
            // Published successfully; drop the now-stale DownloadManager resume record.
            SystemDownload.forget(systemDownloadContext, target.downloadUrl)
        } catch (t: Throwable) {
            // A cancelled caller must propagate — the system download stays alive for resume,
            // but we must not fall through to a duplicate in-app download.
            if (t is kotlinx.coroutines.CancellationException) throw t
            Log.w(
                LOG_TAG,
                "System download failed (${t.message}) - falling back to in-app downloader",
            )
        }
    }

    // internal for testing: the SHA-mismatch path must fail loudly (regression guard).
    internal fun verifyDownloadedFile(target: DownloadTarget) {
        val expectedSize = target.expectedSize
        if ((expectedSize ?: -1L) > 0L && target.targetFile.length() != expectedSize) {
            target.targetFile.delete()
            try { File(target.targetFile.parent, "${target.targetFile.name}.validated").delete() } catch (_: Throwable) {}
            throw IllegalStateException("Downloaded file size mismatch for ${target.modelFile.path}")
        }

        target.modelFile.lfs?.oid?.let { expectedShaValue ->
            val actualSha =
                try {
                    computeSha256(target.targetFile)
                } catch (_: Throwable) {
                    // Size validation above is still a strong signal if hashing is unavailable.
                    null
                }
            if (actualSha != null && !actualSha.equals(expectedShaValue, ignoreCase = true)) {
                target.targetFile.delete()
                try { File(target.targetFile.parent, "${target.targetFile.name}.validated").delete() } catch (_: Throwable) {}
                throw IllegalStateException("Downloaded file sha mismatch for ${target.modelFile.path}")
            }
            if (actualSha != null) {
                writeMarkerFile(target.targetFile, actualSha)
            }
        }
    }

    private fun computeSha256(file: File): String {
        val md = java.security.MessageDigest.getInstance("SHA-256")
        file.inputStream().use { fis ->
            val buffer = ByteArray(128 * 1024)
            var bytesRead = fis.read(buffer)
            while (bytesRead >= 0) {
                md.update(buffer, 0, bytesRead)
                bytesRead = fis.read(buffer)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
