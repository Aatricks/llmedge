package io.aatricks.llmedge.vision

import io.aatricks.llmedge.text.runtime.SmolLM

internal data class VisionPipelineResult(
    val text: String,
    val runtimeMemory: VisionRuntimeMemory,
)

internal class VisionRuntimeExecutor {
    suspend fun execute(
        request: VisionRequest,
        runtime: ManagedVisionRuntime,
        preparedInput: VisionPreparedInput,
        onStatus: ((String) -> Unit)?,
        logStage: (String, String, Long) -> Unit,
        maxTokens: Int = -1,
    ): VisionPipelineResult =
        execute(
            prompt = request.prompt,
            runtime = runtime,
            preparedInput = preparedInput,
            onStatus = onStatus,
            logStage = logStage,
            maxTokens = maxTokens,
        )

    suspend fun execute(
        prompt: String,
        runtime: ManagedVisionRuntime,
        preparedInput: VisionPreparedInput,
        onStatus: ((String) -> Unit)?,
        logStage: (String, String, Long) -> Unit,
        maxTokens: Int = -1,
    ): VisionPipelineResult {
        val smol = runtime.smol
        return when (preparedInput) {
            VisionPreparedInput.PrimedBuffer -> generateText(prompt, smol, onStatus, logStage, maxTokens)
            is VisionPreparedInput.EmbeddingsBuffer -> {
                onStatus?.invoke("Running vision analysis")
                val decodeStartedNs = System.nanoTime()
                val decodeOk = smol.decodeEmbeddingsBuffer(preparedInput.embeddings, nBatch = 1)
                logStage("analyze", "decode_embeddings_buffer", decodeStartedNs)
                check(decodeOk) {
                    "Buffer-based embedding decode failed for the active vision runtime."
                }
                generateText(prompt, smol, onStatus, logStage, maxTokens)
            }

            is VisionPreparedInput.EmbeddingsFile -> {
                onStatus?.invoke("Running vision analysis")
                val decodeStartedNs = System.nanoTime()
                val decodeOk =
                    smol.decodePreparedEmbeddings(
                        preparedInput.embedFile.absolutePath,
                        preparedInput.metaFile.absolutePath,
                        nBatch = 1,
                    )
                logStage("analyze", "decode_embeddings_file", decodeStartedNs)
                check(decodeOk) {
                    "File-based embedding decode failed for the active vision runtime."
                }
                generateText(
                    prompt = prompt,
                    smol = smol,
                    onStatus = onStatus,
                    logStage = logStage,
                    maxTokens = maxTokens,
                )
            }
        }
    }

    private fun generateText(
        prompt: String,
        smol: SmolLM,
        onStatus: ((String) -> Unit)?,
        logStage: (String, String, Long) -> Unit,
        maxTokens: Int,
    ): VisionPipelineResult {
        onStatus?.invoke("Running vision analysis")
        val generationStartedNs = System.nanoTime()
        val response =
            smol.getResponse(
                query = prompt,
                maxTokens = maxTokens,
                batchSize = SmolLM.DEFAULT_BLOCKING_BATCH_SIZE,
            )
        logStage("analyze", "generation", generationStartedNs)
        return VisionPipelineResult(
            text = response,
            runtimeMemory = runtimeMemoryOf(smol),
        )
    }

    private fun runtimeMemoryOf(smol: SmolLM): VisionRuntimeMemory =
        VisionRuntimeMemory(
            nativeBytes = runCatching { smol.getEstimatedNativeMemoryBytes() }.getOrDefault(0L),
            stateBytes = runCatching { smol.getEstimatedStateMemoryBytes() }.getOrDefault(0L),
        )
}
