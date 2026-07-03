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
import io.ktor.http.isSuccess
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.time.LocalDateTime

internal class HFModelSearch(
    private val client: HttpClient,
) {
    @Serializable
    data class ModelSearchResult(
        @SerialName("_id") val idInternal: String,
        val id: String,
        @SerialName("likes") val numLikes: Int,
        @SerialName("downloads") val numDownloads: Int,
        @SerialName("private") val isPrivate: Boolean,
        val tags: List<String>,
        @Serializable(with = CustomDateSerializer::class) val createdAt: LocalDateTime,
        val modelId: String,
    )

    enum class ModelSortParam(val value: String) {
        NONE(""),
        DOWNLOADS("downloads"),
        AUTHOR("author"),
    }

    enum class ModelSearchDirection(val value: Int) {
        ASCENDING(1),
        DESCENDING(-1),
    }

    @Volatile
    private var nextPageUrl: String? = null

    private data class SearchKey(
        val query: String,
        val filter: String,
        val author: String?,
        val sort: ModelSortParam,
        val direction: ModelSearchDirection,
        val limit: Int,
        val full: Boolean,
        val config: Boolean,
    )

    @Volatile
    private var lastSearchKey: SearchKey? = null

    suspend fun searchModels(
        query: String,
        filter: String = "text-generation",
        author: String? = null,
        sort: ModelSortParam = ModelSortParam.DOWNLOADS,
        direction: ModelSearchDirection = ModelSearchDirection.DESCENDING,
        limit: Int = 10,
        full: Boolean = true,
        config: Boolean = true,
    ): List<ModelSearchResult> {
        val currentKey = SearchKey(query, filter, author, sort, direction, limit, full, config)
        if (currentKey != lastSearchKey) {
            nextPageUrl = null
        }
        lastSearchKey = currentKey

        val response =
            nextPageUrl?.let { client.get(it) } ?: run {
                client.get(HFEndpoints.listModelsEndpoint()) {
                    url {
                        parameters.append("search", query)
                        author?.let { parameters.append("author", it) }
                        parameters.append("filter", filter)
                        if (sort != ModelSortParam.NONE) {
                            parameters.append("sort", sort.value)
                        }
                        parameters.append("direction", direction.value.toString())
                        parameters.append("limit", limit.toString())
                        parameters.append("full", full.toString())
                        parameters.append("config", config.toString())
                    }
                }
            }

        if (!response.status.isSuccess()) {
            throw IllegalStateException("Hugging Face API returned error status: ${response.status.value}")
        }

        updatePagination(response.headers["Link"])
        return response.body()
    }

    fun resetPagination() {
        nextPageUrl = null
        lastSearchKey = null
    }

    private fun updatePagination(linkHeader: String?) {
        if (linkHeader == null) {
            nextPageUrl = null
            return
        }
            val regex = """<([^>]+)>;\s*rel=\"([^\"]+)\"""".toRegex()
        val links = regex.findAll(linkHeader).associate { match ->
            val (url, rel) = match.destructured
            rel to url
        }
        nextPageUrl = links["next"]
    }
}
