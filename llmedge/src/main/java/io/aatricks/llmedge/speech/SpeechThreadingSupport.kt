package io.aatricks.llmedge.speech

import io.aatricks.llmedge.runtime.CpuTopology

internal object SpeechThreadingSupport {
    fun resolveThreadCount(requestedThreads: Int): Int =
        if (requestedThreads <= 0) {
            CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING)
        } else {
            requestedThreads
        }
}
