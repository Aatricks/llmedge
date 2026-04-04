package io.aatricks.llmedge.image.diffusion

internal enum class ImageGenerationPhase {
    REQUESTED,
    RUNTIME_ACQUIRED,
    MODEL_READY,
    TXT2IMG_ENTER,
    EXECUTOR_ENTER,
    WAITING_GENERATION_MUTEX,
    JNI_ARGB_ENTER,
    JNI_RGB_ENTER,
    COMPLETED,
    CANCELLED,
    FAILED,
}

internal data class ImageGenerationTraceEvent(
    val requestId: Long,
    val phase: ImageGenerationPhase,
    val detail: String? = null,
    val timestampNanos: Long = System.nanoTime(),
)

internal fun ImageGenerationPhase.isTerminal(): Boolean =
    this == ImageGenerationPhase.COMPLETED ||
        this == ImageGenerationPhase.CANCELLED ||
        this == ImageGenerationPhase.FAILED
