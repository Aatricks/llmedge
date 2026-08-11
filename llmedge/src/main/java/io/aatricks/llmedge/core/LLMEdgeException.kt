package io.aatricks.llmedge.core

import java.io.FileNotFoundException

open class LLMEdgeException(message: String, cause: Throwable? = null) :
    RuntimeException(message, cause)

class NativeBindingException(
    val libraryName: String,
    detail: String,
    cause: Throwable? = null,
) : LLMEdgeException("Failed to load native library '$libraryName': $detail", cause)

class ModelLoadException(
    val modelPath: String,
    detail: String,
    cause: Throwable? = null,
) : LLMEdgeException("Failed to load model at '$modelPath': $detail", cause)

class InferenceFailedException(
    operation: String,
    detail: String,
    cause: Throwable? = null,
) : LLMEdgeException(
    if (detail.isBlank()) "$operation failed" else "$operation failed: $detail",
    cause,
)

class InvalidModelStateException(
    detail: String,
    cause: Throwable? = null,
) : IllegalStateException(detail, cause)

class InvalidGenerationParametersException(detail: String) :
    IllegalArgumentException(detail)

class InsufficientMemoryException(
    val requiredBytes: Long,
    val availableBytes: Long,
    operation: String,
) : LLMEdgeException(
    "Insufficient memory for $operation: requires approximately " +
        "${requiredBytes / (1024L * 1024L)}MB with ${availableBytes / (1024L * 1024L)}MB safely available.",
)

class InvalidModelFileException(
    val modelPath: String,
    detail: String,
    cause: Throwable? = null,
) : LLMEdgeException("Invalid model file '$modelPath': $detail", cause)

class ModelFileNotFoundException(
    val modelPath: String,
    modelKind: String = "Model",
) : FileNotFoundException("$modelKind file not found: $modelPath")

class UnsupportedModelException(detail: String) : UnsupportedOperationException(detail)

/** Failures of the isolated diffusion worker process (DiffusionWorkerMode.ISOLATED_PROCESS). */
open class WorkerProcessException(message: String, cause: Throwable? = null) :
    LLMEdgeException(message, cause)

class WorkerBindException(detail: String, cause: Throwable? = null) :
    WorkerProcessException("Failed to bind the diffusion worker service: $detail", cause)

class WorkerCrashedException(
    val backend: String?,
    val exitReason: Int?,
    /**
     * A short post-mortem recovered on-device (native abort message + signal, or the worker's
     * uncaught-exception stack) so a field crash is diagnosable from the app alone, without adb.
     */
    val crashSummary: String? = null,
) : WorkerProcessException(
    "Diffusion worker process died during generation" +
        (backend?.let { " (backend=$it)" } ?: "") +
        (exitReason?.let { " (exitReason=$it)" } ?: "") +
        (crashSummary?.let { " [$it]" } ?: ""),
)

class WorkerKilledByMemoryException : WorkerProcessException(
    "Diffusion worker process was killed by the low-memory killer. " +
        "Consider forceSequentialLoad, CPU offload options, or a smaller model.",
)

class GenerationHangException(
    val backend: String?,
    val phase: String,
    val stallMs: Long,
    /** True when the total-runtime wall expired rather than the worker going idle. */
    val hardWall: Boolean = false,
) : WorkerProcessException(
    (
        if (hardWall) {
            "Generation exceeded the total time limit (${stallMs}ms elapsed, phase $phase)"
        } else {
            "Generation hung in phase $phase for ${stallMs}ms with an idle worker"
        }
    ) +
        (backend?.let { " (backend=$it)" } ?: "") +
        "; the worker process was killed." +
        if (hardWall) {
            " The device is likely too slow for this model/resolution; try fewer steps or a smaller size."
        } else {
            " This usually indicates a broken GPU driver."
        },
)
