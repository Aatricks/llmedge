package io.aatricks.llmedge.vision

import android.content.Context
import io.aatricks.llmedge.text.runtime.SmolLM

internal data class VisionPipelineResult(
    val text: String,
    val runtimeMemory: VisionRuntimeMemory,
)

internal class VisionRuntimeExecutor(
    private val context: Context,
) {
    suspend fun execute(
        request: VisionRequest,
        runtime: ManagedVisionRuntime,
        preparedInput: VisionPreparedInput,
        onStatus: ((String) -> Unit)?,
        logStage: (String, String, Long) -> Unit,
    ): VisionPipelineResult {
        val smol = runtime.smol
        return when (preparedInput) {
            VisionPreparedInput.PrimedBuffer -> generateText(request.prompt, smol, onStatus, logStage)
            is VisionPreparedInput.EmbeddingsBuffer -> {
                onStatus?.invoke("Running vision analysis")
                val decodeStartedNs = System.nanoTime()
                val decodeOk = smol.decodeEmbeddingsBuffer(preparedInput.embeddings, nBatch = 1)
                logStage("analyze", "decode_embeddings_buffer", decodeStartedNs)
                check(decodeOk) {
                    "Buffer-based embedding decode failed for the active vision runtime."
                }
                generateText(request.prompt, smol, onStatus, logStage)
            }

            is VisionPreparedInput.EmbeddingsFile -> {
                onStatus?.invoke("Running vision analysis")
                val adapter = SmolLMVisionAdapter(context, smol)
                val generationStartedNs = System.nanoTime()
                val analysis =
                    adapter.analyze(
                        ImageSource.FileSource(preparedInput.embedFile),
                        request.prompt,
                        VisionParams(),
                    )
                logStage("analyze", "generation", generationStartedNs)
                VisionPipelineResult(
                    text = analysis.text,
                    runtimeMemory = runtimeMemoryOf(smol),
                )
            }
        }
    }

    private fun generateText(
        prompt: String,
        smol: SmolLM,
        onStatus: ((String) -> Unit)?,
        logStage: (String, String, Long) -> Unit,
    ): VisionPipelineResult {
        onStatus?.invoke("Running vision analysis")
        val generationStartedNs = System.nanoTime()
        val response =
            smol.getResponse(
                query = prompt,
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
            nativeBytes = smol.getEstimatedNativeMemoryBytes(),
            stateBytes = smol.getEstimatedStateMemoryBytes(),
        )
}
