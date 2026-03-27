package io.aatricks.llmedge.core.runtime

internal object BackendFailureClassifier {
    fun isBackendFailure(error: Throwable?): Boolean {
        if (error == null) {
            return false
        }
        return error.message?.contains("backend", ignoreCase = true) == true ||
            error.cause?.message?.contains("backend", ignoreCase = true) == true ||
            error.message?.contains("device lost", ignoreCase = true) == true ||
            error.cause?.message?.contains("device lost", ignoreCase = true) == true
    }
}
