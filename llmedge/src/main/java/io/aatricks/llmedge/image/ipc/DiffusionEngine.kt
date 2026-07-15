package io.aatricks.llmedge.image.ipc

import android.graphics.Bitmap
import io.aatricks.llmedge.image.GenerationStreamEvent
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.image.VideoGenerationRequest
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationTraceEvent

/**
 * Seam between [io.aatricks.llmedge.image.ImageClient]'s public API and the diffusion stack.
 * [InProcessDiffusionEngine] runs the stack in the caller's process (historical behavior);
 * [IsolatedDiffusionEngine] proxies it to the `:llmedge_sd` worker process so that native
 * crashes and GPU-driver hangs cannot take the host app down.
 */
internal interface DiffusionEngine : AutoCloseable {
    suspend fun generate(params: ImageGenerationRequest): Bitmap

    fun generateStream(params: ImageGenerationRequest): kotlinx.coroutines.flow.Flow<GenerationStreamEvent>

    fun generateVideo(params: VideoGenerationRequest): kotlinx.coroutines.flow.Flow<GenerationStreamEvent>

    fun cancelGeneration()

    fun lastGenerationMetrics(): GenerationMetrics?

    fun lastImageRequestTraceForTests(): List<ImageGenerationTraceEvent>
}
