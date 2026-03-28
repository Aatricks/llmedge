package io.aatricks.llmedge.image.diffusion

import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.sync.Mutex

internal class StableDiffusionState(
    val handle: Long,
) {
    val generationMutex: Mutex = Mutex()
    val cancellationRequested: AtomicBoolean = AtomicBoolean(false)
    val rgbBytesThreadLocal: ThreadLocal<ByteArray> = ThreadLocal()

    var modelMetadata: VideoModelMetadata? = null
    var easyCacheSupported: Boolean? = null
    var txt2imgPixelBuffer: IntArray? = null
    var vulkanEnabledForMetrics: Boolean = false

    @Volatile
    var cachedProgressCallback: VideoProgressCallback? = null

    @Volatile
    var lastGenerationMetrics: GenerationMetrics? = null
}
