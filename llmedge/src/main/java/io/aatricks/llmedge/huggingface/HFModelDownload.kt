package io.aatricks.llmedge.huggingface

import io.ktor.client.HttpClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

/**
 * Downloads model files from Hugging Face Hub using OkHttp directly.
 *
 * NOTE: We use OkHttp directly instead of Ktor's HttpClient because Ktor's OkHttp engine
 * buffers the entire response body in Java heap memory before streaming, which causes
 * OutOfMemoryError when downloading large model files (>100MB) on Android devices with
 * limited heap space (typically 256MB).
 *
 * OkHttp's native streaming properly streams the response body directly to disk without
 * buffering the entire content in memory.
 */
internal class HFModelDownload(
    @Suppress("UNUSED_PARAMETER") client: HttpClient, // Kept for API compatibility
) {
    // Use a dedicated OkHttpClient for streaming downloads to avoid heap buffering
    private val downloadClient: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)  // 5 minute read timeout for large files
        .writeTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .retryOnConnectionFailure(true)
        .build()

    suspend fun downloadModelFile(
        modelId: String,
        revision: String,
        filePath: String,
        destination: File,
        token: String?,
        onProgress: ((downloaded: Long, totalBytes: Long?) -> Unit)?,
    ): File = downloadLockFor(destination).withLock {
        withContext(Dispatchers.IO) {
        destination.parentFile?.mkdirs()
        val tempFile = File(destination.parentFile, "${destination.name}.part")
        if (destination.exists()) {
            // Another caller finished this exact download while we waited on the
            // per-destination lock — reuse it instead of re-downloading.
            if (destination.length() > 0L) {
                return@withContext destination
            }
            destination.delete()
        }

        val resumeStart = if (tempFile.exists()) tempFile.length() else 0L

        // Build the download URL
        val url = buildDownloadUrl(modelId, revision, filePath)

        // Build the request with optional resume and auth headers
        val requestBuilder = Request.Builder()
            .url(url)
            .get()

        token?.let { requestBuilder.addHeader("Authorization", "Bearer $it") }
        if (resumeStart > 0L) {
            requestBuilder.addHeader("Range", "bytes=$resumeStart-")
        }

        val request = requestBuilder.build()

        // Execute the request - OkHttp streams directly without buffering in heap
        val response = downloadClient.newCall(request).execute()

        try {
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: ""
                throw IllegalStateException(
                    buildString {
                        append("Failed to download $filePath from $modelId (${response.code})")
                        if (errorBody.isNotBlank()) {
                            append(": ")
                            append(errorBody.take(500)) // Limit error message size
                        }
                    }
                )
            }

            val contentLength = response.header("Content-Length")?.toLongOrNull()
            val expectedLength = contentLength?.let { it + resumeStart } ?: contentLength

            var downloaded = resumeStart

            // If server returned full file (200) despite Range header, start fresh
            if (resumeStart > 0L && response.code == 200) {
                tempFile.delete()
                downloaded = 0L
            }

            onProgress?.invoke(downloaded, expectedLength)

            // Stream directly to file - no heap buffering
            val responseBody = response.body
                ?: throw IllegalStateException("Empty response body for $filePath")

            responseBody.byteStream().use { inputStream ->
                val outputStream = if (resumeStart > 0L && response.code == 206) {
                    FileOutputStream(tempFile, true) // Append mode for resume
                } else {
                    FileOutputStream(tempFile)
                }

                outputStream.use { fos ->
                    val buffer = ByteArray(DOWNLOAD_BUFFER_SIZE)
                    var bytesRead: Int

                    while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                        fos.write(buffer, 0, bytesRead)
                        downloaded += bytesRead
                        onProgress?.invoke(downloaded, expectedLength)
                    }
                    fos.flush()
                }
            }
        } catch (e: OutOfMemoryError) {
            tempFile.delete()
            throw e
        } catch (e: IOException) {
            // Keep the partial file: the next attempt resumes it via the Range header
            // (and a server that ignores Range triggers the fresh-start path above).
            throw e
        } finally {
            response.close()
        }

        if (!tempFile.renameTo(destination)) {
            tempFile.delete()
            throw IllegalStateException("Failed to move temporary file for $filePath")
        }

        destination
        }
    }

    // One mutex per destination path: two concurrent downloads of the same file would
    // otherwise interleave writes into the same .part file and corrupt it.
    private fun downloadLockFor(destination: File): Mutex =
        downloadLocks.computeIfAbsent(destination.absolutePath) { Mutex() }

    private fun buildDownloadUrl(modelId: String, revision: String, filePath: String): String {
        // Encode segments individually so that special characters (including spaces) are encoded
        // correctly, while preserving path segment separators ('/').
        fun encodeSegment(segment: String): String =
            java.net.URLEncoder.encode(segment, "UTF-8").replace("+", "%20")

        val encodedModelId = modelId.trim('/').split('/').joinToString("/") { encodeSegment(it) }
        val encodedRevision = encodeSegment(revision)
        val encodedFilePath = filePath.trim('/').split('/').joinToString("/") { encodeSegment(it) }

        return "https://huggingface.co/$encodedModelId/resolve/$encodedRevision/$encodedFilePath"
    }

    companion object {
        // 64KB buffer for efficient streaming without excessive memory usage
        private const val DOWNLOAD_BUFFER_SIZE = 64 * 1024

        private val downloadLocks = ConcurrentHashMap<String, Mutex>()
    }
}
