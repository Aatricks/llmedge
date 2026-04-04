package io.aatricks.llmedge.text

import io.aatricks.llmedge.TextRuntimeConfig

internal fun resolveBlockingBatchSize(
    config: TextRuntimeConfig,
    requestedBatchSize: Int,
    maxTokens: Int,
): Int {
    val preferredBatchSize =
        when {
            requestedBatchSize == 0 -> config.batchSize
            requestedBatchSize > 0 -> requestedBatchSize
            else -> 1
        }
    return if (maxTokens > 0) {
        minOf(preferredBatchSize, maxTokens.coerceAtLeast(1))
    } else {
        preferredBatchSize
    }
}

internal fun resolveStreamBatchSize(
    config: TextRuntimeConfig,
    requestedBatchSize: Int,
): Int =
    when {
        requestedBatchSize == 0 -> config.streamBatchSize
        requestedBatchSize > 0 -> requestedBatchSize
        else -> 1
    }
