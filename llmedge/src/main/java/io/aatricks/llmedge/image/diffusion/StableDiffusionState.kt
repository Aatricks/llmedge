package io.aatricks.llmedge.image.diffusion

import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.sync.Mutex

internal class StableDiffusionState(
    val handle: Long,
) {
    private val imageTraceLock = Any()

    val generationMutex: Mutex = Mutex()
    val cancellationRequested: AtomicBoolean = AtomicBoolean(false)
    val closed: AtomicBoolean = AtomicBoolean(false)
    val rgbBytesThreadLocal: ThreadLocal<ByteArray> = ThreadLocal()

    var modelMetadata: VideoModelMetadata? = null
    var easyCacheSupported: Boolean? = null
    var txt2imgPixelBuffer: IntArray? = null
    var vulkanEnabledForMetrics: Boolean = false

    @Volatile
    var cachedProgressCallback: VideoProgressCallback? = null

    @Volatile
    var lastGenerationMetrics: GenerationMetrics? = null

    @Volatile
    var currentImageRequestId: Long? = null

    @Volatile
    var currentImagePhase: ImageGenerationPhase? = null

    @Volatile
    var lastImageTrace: List<ImageGenerationTraceEvent> = emptyList()

    private var currentImageTrace: MutableList<ImageGenerationTraceEvent> = mutableListOf()

    fun beginImageTrace(requestId: Long) {
        synchronized(imageTraceLock) {
            currentImageRequestId = requestId
            currentImagePhase = null
            currentImageTrace = mutableListOf()
            lastImageTrace = emptyList()
        }
    }

    fun appendImageTrace(
        phase: ImageGenerationPhase,
        detail: String? = null,
    ): Long? =
        synchronized(imageTraceLock) {
            val requestId = currentImageRequestId ?: return null
            currentImagePhase = phase
            currentImageTrace += ImageGenerationTraceEvent(requestId = requestId, phase = phase, detail = detail)
            if (phase.isTerminal()) {
                lastImageTrace = currentImageTrace.toList()
            }
            requestId
        }

    fun snapshotLastImageTrace(): List<ImageGenerationTraceEvent> =
        synchronized(imageTraceLock) {
            if (lastImageTrace.isNotEmpty()) {
                lastImageTrace.toList()
            } else {
                currentImageTrace.toList()
            }
        }

    fun clearImageTraceState() {
        synchronized(imageTraceLock) {
            if (currentImageTrace.isNotEmpty() && currentImagePhase?.isTerminal() != true) {
                lastImageTrace = currentImageTrace.toList()
            }
            currentImageRequestId = null
            currentImagePhase = null
            currentImageTrace = mutableListOf()
        }
    }
}
