package io.aatricks.llmedge.text.runtime.internal

import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

internal object SmolLMCompletionSupport {
    fun getResponseAsFlow(
        instance: SmolLM,
        query: String,
        dispatcher: CoroutineDispatcher,
        batchSize: Int,
    ): Flow<String> =
        flow {
            val nativePtr = instance.requireLoadedHandle()
            try {
                instance.bridge.startCompletion(instance, nativePtr, query)
                if (batchSize > 1) {
                    val eogBytes = "[EOG]".toByteArray(Charsets.UTF_8)
                    var bytes =
                        try {
                            instance.bridge.completionLoopBatchBytes(
                                instance,
                                nativePtr,
                                batchSize,
                            )
                        } catch (_: Throwable) {
                            null
                        }

                    if (bytes != null) {
                        var chunk: ByteArray = bytes
                        while (!chunk.contentEquals(eogBytes) && chunk.isNotEmpty()) {
                            currentCoroutineContext().ensureActive()
                            emit(String(chunk, Charsets.UTF_8))
                            chunk =
                                instance.bridge.completionLoopBatchBytes(
                                    instance,
                                    nativePtr,
                                    batchSize,
                                ) ?: break
                        }
                    } else {
                        var piece =
                            instance.bridge.completionLoopBatch(
                                instance,
                                nativePtr,
                                batchSize,
                            )
                        while (piece != "[EOG]" && piece.isNotEmpty()) {
                            currentCoroutineContext().ensureActive()
                            emit(piece)
                            piece =
                                instance.bridge.completionLoopBatch(
                                    instance,
                                    nativePtr,
                                    batchSize,
                                )
                        }
                    }
                } else {
                    var piece = instance.bridge.completionLoop(instance, nativePtr)
                    while (piece != "[EOG]") {
                        currentCoroutineContext().ensureActive()
                        emit(piece)
                        piece = instance.bridge.completionLoop(instance, nativePtr)
                    }
                }
            } catch (e: IllegalStateException) {
                throw InferenceFailedException(
                    operation = "SmolLM streaming completion",
                    detail = e.message ?: "The native completion loop failed.",
                    cause = e,
                )
            } finally {
                instance.bridge.stopCompletion(instance, nativePtr)
            }
        }.flowOn(dispatcher)

    fun getResponse(
        instance: SmolLM,
        query: String,
        maxTokens: Int,
        batchSize: Int,
    ): String {
        val nativePtr = instance.requireLoadedHandle()
        SmolLM.logDebug(
            "getResponse: starting completion. maxTokens=$maxTokens, batchSize=$batchSize, queryLength=${query.length}",
        )
        instance.bridge.startCompletion(instance, nativePtr, query)
        try {
            val estimatedCapacity = if (maxTokens > 0) maxTokens * 4 else 512
            val responseBuilder = StringBuilder(estimatedCapacity)
            var tokensGenerated = 0

            if (batchSize > 1) {
                val effectiveBatch = if (maxTokens > 0) minOf(batchSize, maxTokens) else batchSize
                var piece =
                    instance.bridge.completionLoopBatch(
                        instance,
                        nativePtr,
                        effectiveBatch,
                    )
                while (piece != "[EOG]" && piece.isNotEmpty()) {
                    responseBuilder.append(piece)
                    tokensGenerated += effectiveBatch

                    if (maxTokens > 0 && tokensGenerated >= maxTokens) {
                        SmolLM.logDebug("getResponse: maxTokens ($maxTokens) reached. Stopping.")
                        break
                    }

                    val remaining =
                        if (maxTokens > 0) minOf(batchSize, maxTokens - tokensGenerated)
                        else batchSize
                    if (remaining <= 0) break
                    piece =
                        instance.bridge.completionLoopBatch(
                            instance,
                            nativePtr,
                            remaining,
                        )
                }
                if (piece == "[EOG]") {
                    SmolLM.logDebug(
                        "getResponse: [EOG] received after ~$tokensGenerated tokens.",
                    )
                }
            } else {
                var piece = instance.bridge.completionLoop(instance, nativePtr)
                while (piece != "[EOG]") {
                    responseBuilder.append(piece)
                    tokensGenerated++

                    if (tokensGenerated % 10 == 0) {
                        SmolLM.logDebug("Generated $tokensGenerated tokens...")
                    }

                    if (maxTokens > 0 && tokensGenerated >= maxTokens) {
                        SmolLM.logDebug("getResponse: maxTokens ($maxTokens) reached. Stopping.")
                        break
                    }

                    piece = instance.bridge.completionLoop(instance, nativePtr)
                }
                if (piece == "[EOG]") {
                    SmolLM.logDebug(
                        "getResponse: [EOG] received after $tokensGenerated tokens.",
                    )
                }
            }

            return responseBuilder.toString().also { response ->
                SmolLM.logDebug("getResponse: finished. Total length=${response.length}")
            }
        } catch (e: IllegalStateException) {
            throw InferenceFailedException(
                operation = "SmolLM completion",
                detail = e.message ?: "The native completion loop failed.",
                cause = e,
            )
        } finally {
            instance.bridge.stopCompletion(instance, nativePtr)
        }
    }

    fun stopCompletion(instance: SmolLM) {
        val nativePtr = instance.state.nativePtr
        if (nativePtr == 0L) {
            return
        }
        SmolLM.logDebug("stopCompletion invoked")
        try {
            instance.bridge.stopCompletion(instance, nativePtr)
        } catch (e: Throwable) {
            SmolLM.logWarning("stopCompletion failed: ${e.message}")
        }
    }
}
