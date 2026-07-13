/*
 * Copyright (C) 2025 Shubham Panchal
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

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.http.HttpHeaders
import io.ktor.http.isSuccess
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

internal class HFModelTree(
    private val client: HttpClient,
) {
    @Serializable
    data class HFModelFile(
        val type: String? = null,
        val oid: String? = null,
        val size: Long? = null,
        @SerialName("path") val pathStr: String? = null,
        @SerialName("rfilename") val rfilenameStr: String? = null,
        @SerialName("lfs") val lfs: LfsMetadata? = null,
    ) {
        constructor(
            type: String? = null,
            oid: String? = null,
            size: Long? = null,
            path: String,
            lfs: LfsMetadata? = null,
        ) : this(
            type = type,
            oid = oid,
            size = size,
            pathStr = path,
            rfilenameStr = null,
            lfs = lfs
        )

        val path: String
            get() = pathStr ?: rfilenameStr ?: ""

        @Serializable
        data class LfsMetadata(
            val oid: String,
            val size: Long,
        )
    }

    suspend fun getModelFileTree(
        modelId: String,
        revision: String,
        token: String? = null,
        recursive: Boolean = false,
    ): List<HFModelFile> {
        // The HF tree API lists only the top level by default, so a nested filename such as
        // "minit2i-b-16/transformer/diffusion_pytorch_model.safetensors" never appears as a
        // selection candidate (the subdirectory shows up as a "directory" entry instead).
        // Request the recursive listing when resolving an exact repo file that may live in a
        // subdirectory. (For very large repos HF paginates the recursive tree; the models we
        // resolve this way are small, so a single page suffices.)
        val endpoint = HFEndpoints.modelTreeEndpoint(modelId, revision)
        val url = if (recursive) "$endpoint?recursive=true" else endpoint
        val response =
            client.get(url) {
                token?.let { header(HttpHeaders.Authorization, "Bearer $it") }
            }
        if (!response.status.isSuccess()) {
            throw IllegalArgumentException("Hugging Face model '$modelId' not found")
        }
        return response.body()
    }
}
