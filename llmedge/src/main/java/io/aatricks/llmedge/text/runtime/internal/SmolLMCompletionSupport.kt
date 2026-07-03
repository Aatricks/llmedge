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
    private const val EOG = "[EOG]"
    private const val MAX_CONSECUTIVE_EMPTY_BATCHES = 4

    /**
     * Fetches completion pieces, preferring the byte-array native path: generated text can
     * contain 4-byte UTF-8 sequences (emoji), which are invalid Modified UTF-8 and unsafe to
     * return through `NewStringUTF`. Falls back to the legacy String natives only when the
     * loaded library predates the bytes entry point.
     *
     * Every native call is made under [SmolLM.nativeCallLock] with the handle re-read inside
     * the lock, so a concurrent `close()` results in an IllegalStateException on the next
     * step instead of a native use-after-free.
     */
    private class CompletionStepper(private val instance: SmolLM) {
        private var bytesSupported = true
        private var consecutiveEmptyBatches = 0

        /**
         * Returns the next piece, or null when the stream is over. An empty string
         * means "keep polling" (the native side is buffering an incomplete UTF-8
         * sequence) — except on the legacy String batch path, where empty has always
         * meant end-of-stream and is mapped to null to preserve that contract.
         */
        fun next(count: Int): String? {
            synchronized(instance.nativeCallLock) {
                val nativePtr = instance.requireLoadedHandle()
                if (bytesSupported) {
                    try {
                        val bytes =
                            instance.bridge.completionLoopBatchBytes(instance, nativePtr, count)
                        if (bytes != null) {
                            val piece = String(bytes, Charsets.UTF_8)
                            return if (piece == EOG) null else piece
                        }
                        // null = this bridge has no bytes path (contract default / old lib);
                        // fall through to the String natives below for this and later calls.
                        bytesSupported = false
                    } catch (_: UnsatisfiedLinkError) {
                        bytesSupported = false
                    }
                }
                if (count > 1) {
                    val piece = instance.bridge.completionLoopBatch(instance, nativePtr, count)
                    if (piece == EOG) return null
                    // The real native returns "" mid-generation while it buffers an
                    // incomplete UTF-8 sequence (EOG is always the explicit marker),
                    // but legacy/fake bridges return "" at end-of-stream. A buffered
                    // sequence resolves within a few tokens, so poll a bounded number
                    // of consecutive empties before treating it as the end.
                    if (piece.isEmpty()) {
                        return if (++consecutiveEmptyBatches >= MAX_CONSECUTIVE_EMPTY_BATCHES) null else piece
                    }
                    consecutiveEmptyBatches = 0
                    return piece
                }
                val piece = instance.bridge.completionLoop(instance, nativePtr)
                return if (piece == EOG) null else piece
            }
        }
    }

    private fun startCompletionLocked(instance: SmolLM, query: String) {
        instance.applyInferenceThreadAffinity()
        synchronized(instance.nativeCallLock) {
            instance.bridge.startCompletion(instance, instance.requireLoadedHandle(), query)
        }
    }

    private fun stopCompletionLocked(instance: SmolLM) {
        synchronized(instance.nativeCallLock) {
            val nativePtr = instance.state.nativePtr
            if (nativePtr != 0L) {
                instance.bridge.stopCompletion(instance, nativePtr)
            }
        }
    }

    fun getResponseAsFlow(
        instance: SmolLM,
        query: String,
        dispatcher: CoroutineDispatcher,
        batchSize: Int,
    ): Flow<String> =
        flow {
            try {
                startCompletionLocked(instance, query)
                val stepper = CompletionStepper(instance)
                val step = batchSize.coerceAtLeast(1)
                while (true) {
                    currentCoroutineContext().ensureActive()
                    val piece = stepper.next(step) ?: break
                    // An empty piece means the native side is buffering an incomplete
                    // UTF-8 sequence; keep looping until it resolves or the stream ends.
                    if (piece.isNotEmpty()) {
                        emit(piece)
                    }
                }
            } catch (e: IllegalStateException) {
                throw InferenceFailedException(
                    operation = "SmolLM streaming completion",
                    detail = e.message ?: "The native completion loop failed.",
                    cause = e,
                )
            } finally {
                stopCompletionLocked(instance)
            }
        }.flowOn(dispatcher)

    fun getResponse(
        instance: SmolLM,
        query: String,
        maxTokens: Int,
        batchSize: Int,
    ): String {
        SmolLM.logDebug(
            "getResponse: starting completion. maxTokens=$maxTokens, batchSize=$batchSize, queryLength=${query.length}",
        )
        startCompletionLocked(instance, query)
        try {
            val estimatedCapacity = if (maxTokens > 0) maxTokens * 4 else 512
            val responseBuilder = StringBuilder(estimatedCapacity)
            val stepper = CompletionStepper(instance)
            var tokensGenerated = 0

            while (true) {
                val step =
                    if (maxTokens > 0) {
                        minOf(batchSize.coerceAtLeast(1), maxTokens - tokensGenerated)
                    } else {
                        batchSize.coerceAtLeast(1)
                    }
                if (step <= 0) {
                    SmolLM.logDebug("getResponse: maxTokens ($maxTokens) reached. Stopping.")
                    break
                }
                val piece = stepper.next(step)
                if (piece == null) {
                    SmolLM.logDebug("getResponse: stream ended after ~$tokensGenerated tokens.")
                    break
                }
                responseBuilder.append(piece)
                tokensGenerated += step

                if (tokensGenerated % 40 == 0) {
                    SmolLM.logDebug("Generated ~$tokensGenerated tokens...")
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
            stopCompletionLocked(instance)
        }
    }

    fun stopCompletion(instance: SmolLM) {
        SmolLM.logDebug("stopCompletion invoked")
        try {
            stopCompletionLocked(instance)
        } catch (e: Throwable) {
            SmolLM.logWarning("stopCompletion failed: ${e.message}")
        }
    }
}
