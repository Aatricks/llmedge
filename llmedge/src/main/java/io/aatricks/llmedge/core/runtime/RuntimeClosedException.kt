package io.aatricks.llmedge.core.runtime

/**
 * Exception thrown when a runtime is accessed after it has been closed.
 */
class RuntimeClosedException(message: String) : IllegalStateException(message)
