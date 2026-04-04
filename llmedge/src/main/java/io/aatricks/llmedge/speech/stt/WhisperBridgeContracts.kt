package io.aatricks.llmedge.speech.stt

import io.aatricks.llmedge.runtime.ComputeBackend

internal interface WhisperNativeBridgeContract {
    fun transcribe(
        handle: Long,
        samples: FloatArray,
        params: Whisper.TranscribeParams,
        progressCallback: Whisper.ProgressCallback?,
        segmentCallback: Whisper.SegmentCallback?,
    ): Array<Whisper.TranscriptionSegment>?

    fun detectLanguage(
        handle: Long,
        samples: FloatArray,
        nThreads: Int,
    ): Int

    fun getFullText(handle: Long): String

    fun close(handle: Long)
}

internal interface WhisperLoadBridgeContract {
    fun create(
        modelPath: String,
        backend: ComputeBackend,
        flashAttn: Boolean,
        gpuDevice: Int,
    ): Long
}
