package io.aatricks.llmedge.image

import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationTraceEvent
import io.aatricks.llmedge.image.diffusion.StableDiffusion

internal class ImageClientState {
    @Volatile
    var lastGenerationMetrics: GenerationMetrics? = null

    @Volatile
    var lastImageRequestTrace: List<ImageGenerationTraceEvent> = emptyList()

    @Volatile
    var activeModel: StableDiffusion? = null

    fun resetForRequest() {
        lastGenerationMetrics = null
        lastImageRequestTrace = emptyList()
    }
}
