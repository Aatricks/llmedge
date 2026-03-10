package io.aatricks.llmedge.core

import io.aatricks.llmedge.model.ModelSpec

sealed interface ProgressEvent {
    data class Downloading(
        val model: ModelSpec,
        val downloadedBytes: Long,
        val totalBytes: Long?,
    ) : ProgressEvent

    data class Status(val message: String) : ProgressEvent

    data class Step(
        val message: String,
        val current: Int,
        val total: Int,
    ) : ProgressEvent
}
