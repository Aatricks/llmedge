package io.aatricks.llmedge.core.runtime

internal object BackendFailureClassifier {
    fun isBackendFailure(error: Throwable?): Boolean {
        if (error == null) {
            return false
        }
        return matches(error.message) || matches(error.cause?.message)
    }

    private fun matches(message: String?): Boolean {
        if (message.isNullOrBlank()) {
            return false
        }
        val normalized = message.lowercase().filter(Char::isLetter)
        return "backend" in message.lowercase() || "devicelost" in normalized
    }
}
