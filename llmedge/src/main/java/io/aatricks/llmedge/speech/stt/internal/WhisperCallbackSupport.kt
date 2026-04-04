package io.aatricks.llmedge.speech.stt.internal

import io.aatricks.llmedge.speech.stt.Whisper

internal object WhisperCallbackSupport {
    fun progressCallbackBridge(callback: Whisper.ProgressCallback): Any =
        object : Any() {
            @Suppress("unused")
            fun onProgress(progress: Int) {
                callback.onProgress(progress)
            }
        }

    fun segmentCallbackBridge(callback: Whisper.SegmentCallback): Any =
        object : Any() {
            @Suppress("unused")
            fun onNewSegment(index: Int, startTime: Long, endTime: Long, text: String) {
                callback.onNewSegment(index, startTime, endTime, text)
            }
        }
}
