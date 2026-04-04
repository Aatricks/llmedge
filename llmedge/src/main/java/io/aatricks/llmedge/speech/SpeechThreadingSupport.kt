package io.aatricks.llmedge.speech

internal object SpeechThreadingSupport {
    fun resolveThreadCount(requestedThreads: Int): Int =
        if (requestedThreads <= 0) {
            Runtime.getRuntime().availableProcessors().coerceAtMost(8)
        } else {
            requestedThreads
        }
}
