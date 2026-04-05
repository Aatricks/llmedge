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
import java.io.File

/**
 * Advanced helper to discover and download artifacts from Hugging Face.
 *
 * Most application code should prefer `LLMEdge.create(...).models` or `ModelRepository`. Use this
 * object directly only when you need low-level control over artifact selection, direct cache
 * inspection, or custom download policy outside the facade-managed path.
 */
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
    ): ModelDownloadResult {
        val downloadContext = HFArtifactRequestSupport.artifactDownloadContext(context, preferSystemDownloader)
        return ensureModelOnDisk(
            destinationRoot = downloadContext.destinationRoot,
            modelId = modelId,
            revision = revision,
            preferredQuantizations = preferredQuantizations,
            filename = filename,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            systemDownloadContext = downloadContext.systemDownloadContext,
            onProgress = onProgress,
        )
    }

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
    ): ModelDownloadResult = ensureArtifactOnDisk(
        destinationRoot = destinationRoot,
        request = HFArtifactRequestSupport.modelArtifactRequest(
            modelId = modelId,
            revision = revision,
            preferredQuantizations = preferredQuantizations,
            filename = filename,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            onProgress = onProgress,
        ),
        systemDownloadContext = systemDownloadContext,
    )

    suspend fun ensureWanAssetsOnDisk(
        context: Context,
        wanModelId: String,
        preferSystemDownloader: Boolean = true,
        token: String? = null,
        forceDownload: Boolean = false,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): Triple<ModelDownloadResult, ModelDownloadResult?, ModelDownloadResult?> =
        HFWanAssetSupport.ensureWanAssetsOnDisk(
            context = context,
            wanModelId = wanModelId,
            preferSystemDownloader = preferSystemDownloader,
            token = token,
            forceDownload = forceDownload,
            onProgress = onProgress,
        )

    fun clearCache(context: Context) {
        val root = HFDownloadSupport.defaultModelsRoot(context)
        if (root.exists()) {
            root.deleteRecursively()
        }
    }

    fun listCachedModels(context: Context): List<File> {
        val root = HFDownloadSupport.defaultModelsRoot(context)
        return if (root.exists() && root.isDirectory) {
            root.listFiles()?.filter { it.isDirectory }?.toList() ?: emptyList()
        } else {
            emptyList()
        }
    }

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
    ): ModelDownloadResult {
        val downloadContext = HFArtifactRequestSupport.artifactDownloadContext(context, preferSystemDownloader)
        return ensureRepoFileOnDisk(
            destinationRoot = downloadContext.destinationRoot,
            modelId = modelId,
            revision = revision,
            filename = filename,
            allowedExtensions = allowedExtensions,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            systemDownloadContext = downloadContext.systemDownloadContext,
            onProgress = onProgress,
        )
    }

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
    ): ModelDownloadResult = ensureArtifactOnDisk(
        destinationRoot = destinationRoot,
        request = HFArtifactRequestSupport.repoFileArtifactRequest(
            modelId = modelId,
            revision = revision,
            filename = filename,
            allowedExtensions = allowedExtensions,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            onProgress = onProgress,
        ),
        systemDownloadContext = systemDownloadContext,
    )

    internal suspend fun ensureArtifactOnDisk(
        context: Context,
        request: ArtifactRequest,
    ): ModelDownloadResult {
        val downloadContext = HFArtifactRequestSupport.artifactDownloadContext(context, request.preferSystemDownloader)
        return ensureArtifactOnDisk(
            destinationRoot = downloadContext.destinationRoot,
            request = request,
            systemDownloadContext = downloadContext.systemDownloadContext,
        )
    }

    internal suspend fun ensureArtifactOnDisk(
        destinationRoot: File,
        request: ArtifactRequest,
        systemDownloadContext: Context? = null,
    ): ModelDownloadResult =
        when (request) {
            is ArtifactRequest.Model ->
                HFDownloadSupport.ensureFileOnDisk(
                    destinationRoot = destinationRoot,
                    modelId = request.modelId,
                    revision = request.revision,
                    token = request.token,
                    forceDownload = request.forceDownload,
                    preferSystemDownloader = request.preferSystemDownloader,
                    systemDownloadContext = systemDownloadContext,
                    onProgress = request.onProgress,
                    noMatchMessage = "No GGUF file found for '${request.modelId}' (revision '${request.revision}')",
                    fileSelector = { files ->
                        HFFileSelectionSupport.selectModelFile(files, request.filename, request.preferredQuantizations)
                    },
                )

            is ArtifactRequest.RepoFile ->
                HFDownloadSupport.ensureFileOnDisk(
                    destinationRoot = destinationRoot,
                    modelId = request.modelId,
                    revision = request.revision,
                    token = request.token,
                    forceDownload = request.forceDownload,
                    preferSystemDownloader = request.preferSystemDownloader,
                    systemDownloadContext = systemDownloadContext,
                    onProgress = request.onProgress,
                    noMatchMessage = "No file found for '${request.modelId}' matching ${request.filename ?: request.allowedExtensions}",
                    fileSelector = { files ->
                        HFFileSelectionSupport.selectRepoFile(files, request.filename, request.allowedExtensions)
                    },
                )
        }

    val DEFAULT_QUANTIZATION_PRIORITIES: List<String> =
        HFFileSelectionSupport.DEFAULT_QUANTIZATION_PRIORITIES

    fun sanitize(modelId: String): String = HFFileSelectionSupport.sanitize(modelId)

    internal fun isFileValidCached(targetFile: File, expectedSize: Long?, expectedSha: String?): Boolean =
        HFDownloadSupport.isFileValidCached(targetFile, expectedSize, expectedSha)
}
