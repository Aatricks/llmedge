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
