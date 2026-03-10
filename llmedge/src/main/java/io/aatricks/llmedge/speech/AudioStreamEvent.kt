package io.aatricks.llmedge.speech

import io.aatricks.llmedge.speech.tts.BarkTTS

sealed interface AudioStreamEvent {
    data object Started : AudioStreamEvent
    data class Progress(
        val step: BarkTTS.EncodingStep,
        val percent: Int,
    ) : AudioStreamEvent
    data class Result(val audio: BarkTTS.AudioResult) : AudioStreamEvent
    data object Completed : AudioStreamEvent
}
