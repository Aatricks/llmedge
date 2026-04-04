package io.aatricks.llmedge.speech.stt.internal

import io.aatricks.llmedge.speech.stt.Whisper

internal object WhisperSubtitleSupport {
    fun toSrtEntry(segment: Whisper.TranscriptionSegment): String {
        val startFormatted = formatTimeSrt(segment.startTimeMs)
        val endFormatted = formatTimeSrt(segment.endTimeMs)
        return "${segment.index + 1}\n$startFormatted --> $endFormatted\n${segment.text}\n"
    }

    fun toVttEntry(segment: Whisper.TranscriptionSegment): String {
        val startFormatted = formatTimeVtt(segment.startTimeMs)
        val endFormatted = formatTimeVtt(segment.endTimeMs)
        return "$startFormatted --> $endFormatted\n${segment.text}\n"
    }

    fun generateSrt(segments: List<Whisper.TranscriptionSegment>): String =
        segments.joinToString("\n", transform = ::toSrtEntry)

    fun generateVtt(segments: List<Whisper.TranscriptionSegment>): String =
        "WEBVTT\n\n" + segments.joinToString("\n", transform = ::toVttEntry)

    private fun formatTimeSrt(milliseconds: Long): String {
        val hours = milliseconds / 3600000
        val minutes = (milliseconds % 3600000) / 60000
        val seconds = (milliseconds % 60000) / 1000
        val millis = milliseconds % 1000
        return String.format("%02d:%02d:%02d,%03d", hours, minutes, seconds, millis)
    }

    private fun formatTimeVtt(milliseconds: Long): String {
        val hours = milliseconds / 3600000
        val minutes = (milliseconds % 3600000) / 60000
        val seconds = (milliseconds % 60000) / 1000
        val millis = milliseconds % 1000
        return String.format("%02d:%02d:%02d.%03d", hours, minutes, seconds, millis)
    }
}
