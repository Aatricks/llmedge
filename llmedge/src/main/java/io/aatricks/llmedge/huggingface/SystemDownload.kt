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

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

internal object SystemDownload {

    /**
     * DownloadManager runs in the system process, so an enqueued download keeps making progress
     * after this app is backgrounded or killed. We use DownloadManager itself as the resume
     * registry: [COLUMN_URI][DownloadManager.COLUMN_URI] records the source URL and
     * [COLUMN_LOCAL_URI][DownloadManager.COLUMN_LOCAL_URI] records the destination, both surviving
     * process death. On a retry we look up any prior download for the same URL and reconnect to it
     * instead of starting over.
     */
    enum class ResumeAction { FINALIZE, MONITOR, REENQUEUE }

    /** A prior DownloadManager download discovered by source URL. */
    data class ExistingDownload(
        val id: Long,
        val status: Int,
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val localPath: String?,
    )

    suspend fun download(
        context: Context,
        url: String,
        token: String?,
        destination: File,
        displayName: String,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)?,
    ): File {
        val downloadManager = context.getSystemService(DownloadManager::class.java)
            ?: throw IllegalStateException("DownloadManager service not available")

        val existing = findExistingByUrl(downloadManager, url)
        if (existing != null) {
            val localFile = existing.localPath?.let(::File)
            when (decideResume(existing, localFile?.length() ?: 0L)) {
                ResumeAction.FINALIZE -> {
                    // A prior run already finished this download; adopt its file (the SHA gate in
                    // HFDownloadSupport.verifyDownloadedFile still validates it before publishing).
                    return localFile!!
                }
                ResumeAction.MONITOR -> {
                    // Still in flight in the system process; reconnect to the same download.
                    return monitor(downloadManager, existing.id, localFile ?: destination, onProgress)
                }
                ResumeAction.REENQUEUE -> {
                    // Failed or gone stale; discard it and start fresh.
                    downloadManager.remove(existing.id)
                }
            }
        }

        destination.parentFile?.mkdirs()
        if (destination.exists()) {
            destination.delete()
        }

        val request = DownloadManager.Request(Uri.parse(url))
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(true)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationUri(Uri.fromFile(destination))
            .setTitle(displayName)

        token?.let { request.addRequestHeader("Authorization", "Bearer $it") }

        val downloadId = downloadManager.enqueue(request)
        return monitor(downloadManager, downloadId, destination, onProgress)
    }

    /**
     * Watches [downloadId] to completion. On coroutine cancellation the monitor detaches but the
     * download is deliberately left running in the system process so it can be resumed later — the
     * partial file is never deleted here.
     */
    private suspend fun monitor(
        downloadManager: DownloadManager,
        downloadId: Long,
        destination: File,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)?,
    ): File {
        val query = DownloadManager.Query().setFilterById(downloadId)
        return suspendCancellableCoroutine { cont ->
            val monitorScope = CoroutineScope(Dispatchers.IO + Job())
            val monitorJob = monitorScope.launch {
                while (isActive && cont.isActive) {
                    try {
                        downloadManager.query(query).use { cursor ->
                            if (!cursor.moveToFirst()) {
                                return@use
                            }
                            val status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                            val downloaded = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                            val total = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                            onProgress?.invoke(downloaded, if (total > 0) total else null)

                            when (status) {
                                DownloadManager.STATUS_SUCCESSFUL -> {
                                    if (cont.isActive) {
                                        cont.resume(destination)
                                    }
                                    cancel()
                                    return@launch
                                }
                                DownloadManager.STATUS_FAILED -> {
                                    val reason = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
                                    if (cont.isActive) {
                                        cont.resumeWithException(IllegalStateException("System download failed (reason=$reason)"))
                                    }
                                    cancel()
                                    return@launch
                                }
                                else -> {
                                    if (hasCompletePayload(destination, downloaded, total) && cont.isActive) {
                                        cont.resume(destination)
                                        cancel()
                                        return@launch
                                    }
                                }
                            }
                        }
                    } catch (t: Throwable) {
                        if (cont.isActive) {
                            cont.resumeWithException(t)
                        }
                        cancel()
                        return@launch
                    }
                    delay(500L)
                }
            }

            cont.invokeOnCancellation {
                // Detach only. The system download keeps running so leaving the screen or the app
                // does not abort a multi-GB transfer; a later retry reconnects via findExistingByUrl.
                monitorJob.cancel()
                monitorScope.cancel()
            }
        }
    }

    /**
     * Decides how to treat a prior download found for the same URL. Pure so it can be unit-tested
     * without a DownloadManager. [localLength] is the current size of [ExistingDownload.localPath].
     */
    internal fun decideResume(existing: ExistingDownload, localLength: Long): ResumeAction =
        when (existing.status) {
            DownloadManager.STATUS_SUCCESSFUL ->
                // Only adopt a finished download whose bytes are actually complete on disk; a
                // SUCCESSFUL status with a truncated/missing file must not be trusted.
                if (existing.localPath != null &&
                    existing.totalBytes > 0L &&
                    localLength >= existing.totalBytes
                ) {
                    ResumeAction.FINALIZE
                } else {
                    ResumeAction.REENQUEUE
                }
            DownloadManager.STATUS_RUNNING,
            DownloadManager.STATUS_PENDING,
            DownloadManager.STATUS_PAUSED ->
                if (existing.localPath != null) ResumeAction.MONITOR else ResumeAction.REENQUEUE
            else -> ResumeAction.REENQUEUE // STATUS_FAILED or unknown
        }

    /**
     * Orders two same-URL downloads so [findExistingByUrl] is independent of cursor order: a
     * resumable (non-failed) download beats a failed one, then more bytes downloaded wins.
     */
    internal fun isBetterMatch(candidate: ExistingDownload, current: ExistingDownload): Boolean {
        val candidateResumable = candidate.status != DownloadManager.STATUS_FAILED
        val currentResumable = current.status != DownloadManager.STATUS_FAILED
        if (candidateResumable != currentResumable) return candidateResumable
        return candidate.bytesDownloaded > current.bytesDownloaded
    }

    /** Finds the best resumable DownloadManager download whose source URL matches [url]. */
    private fun findExistingByUrl(downloadManager: DownloadManager, url: String): ExistingDownload? {
        return try {
            downloadManager.query(DownloadManager.Query()).use { cursor ->
                val uriCol = cursor.getColumnIndex(DownloadManager.COLUMN_URI)
                val idCol = cursor.getColumnIndex(DownloadManager.COLUMN_ID)
                val statusCol = cursor.getColumnIndex(DownloadManager.COLUMN_STATUS)
                val soFarCol = cursor.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
                val totalCol = cursor.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
                val localCol = cursor.getColumnIndex(DownloadManager.COLUMN_LOCAL_URI)
                if (uriCol < 0 || idCol < 0 || statusCol < 0) return null
                var match: ExistingDownload? = null
                while (cursor.moveToNext()) {
                    if (cursor.getString(uriCol) != url) continue
                    val candidate = ExistingDownload(
                        id = cursor.getLong(idCol),
                        status = cursor.getInt(statusCol),
                        bytesDownloaded = if (soFarCol >= 0) cursor.getLong(soFarCol) else 0L,
                        totalBytes = if (totalCol >= 0) cursor.getLong(totalCol) else 0L,
                        localPath = if (localCol >= 0) cursor.getString(localCol)?.let { Uri.parse(it).path } else null,
                    )
                    // Cursor order is not guaranteed, so pick deterministically: prefer a resumable
                    // (non-failed) download, and among those the one furthest along.
                    if (match == null || isBetterMatch(candidate, match!!)) {
                        match = candidate
                    }
                }
                match
            }
        } catch (_: Throwable) {
            null
        }
    }

    /**
     * Destination paths of DownloadManager downloads that are still resumable (in flight or
     * finished-but-unpublished). [HFDownloadSupport] excludes these when sweeping orphaned temp
     * files so it never deletes a partial we intend to resume.
     */
    internal fun resumableTempPaths(context: Context): Set<String> {
        val downloadManager = context.getSystemService(DownloadManager::class.java) ?: return emptySet()
        return try {
            downloadManager.query(DownloadManager.Query()).use { cursor ->
                val statusCol = cursor.getColumnIndex(DownloadManager.COLUMN_STATUS)
                val localCol = cursor.getColumnIndex(DownloadManager.COLUMN_LOCAL_URI)
                if (statusCol < 0 || localCol < 0) return emptySet()
                val paths = mutableSetOf<String>()
                while (cursor.moveToNext()) {
                    val status = cursor.getInt(statusCol)
                    if (status == DownloadManager.STATUS_FAILED) continue
                    val path = cursor.getString(localCol)?.let { Uri.parse(it).path } ?: continue
                    paths.add(path)
                }
                paths
            }
        } catch (_: Throwable) {
            emptySet()
        }
    }

    /**
     * Drops the DownloadManager record for [url] once its file has been published. The file has
     * already been moved out of the download location, so this only clears the (now stale) record.
     */
    internal fun forget(context: Context, url: String) {
        val downloadManager = context.getSystemService(DownloadManager::class.java) ?: return
        val existing = findExistingByUrl(downloadManager, url) ?: return
        try {
            downloadManager.remove(existing.id)
        } catch (_: Throwable) {
            // best-effort cleanup of the registry
        }
    }

    internal fun hasCompletePayload(
        destination: File,
        downloadedBytes: Long,
        totalBytes: Long,
    ): Boolean =
        totalBytes > 0L &&
            downloadedBytes >= totalBytes &&
            destination.isFile &&
            destination.length() == totalBytes
}
