package io.aatricks.llmedge.speech.stt.internal

import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.speech.SpeechThreadingSupport
import io.aatricks.llmedge.speech.stt.Whisper

internal object WhisperInferenceOperations {
    fun transcribe(
        samples: FloatArray,
        params: Whisper.TranscribeParams,
        transcribeNative: (
            samples: FloatArray,
            nThreads: Int,
            translate: Boolean,
            language: String?,
            detectLanguage: Boolean,
            tokenTimestamps: Boolean,
            maxLen: Int,
            splitOnWord: Boolean,
            temperature: Float,
            beamSize: Int,
            suppressBlank: Boolean,
            printProgress: Boolean,
        ) -> Array<Whisper.TranscriptionSegment>?,
    ): List<Whisper.TranscriptionSegment> {
        require(samples.isNotEmpty()) { "Audio samples cannot be empty" }

        val effectiveThreads = SpeechThreadingSupport.resolveThreadCount(params.nThreads)
        val segments =
            transcribeNative(
                samples,
                effectiveThreads,
                params.translate,
                params.language,
                params.detectLanguage,
                params.tokenTimestamps,
                params.maxLen,
                params.splitOnWord,
                params.temperature,
                params.beamSize,
                params.suppressBlank,
                params.printProgress,
            )
                ?: throw InferenceFailedException(
                    operation = "Whisper transcription",
                    detail = "The native transcription call returned no segments.",
                )

        return segments.toList()
    }

    fun detectLanguage(
        samples: FloatArray,
        nThreads: Int,
        detectLanguageNative: (samples: FloatArray, nThreads: Int) -> Int,
        resolveLanguageString: (Int) -> String,
    ): String? {
        require(samples.isNotEmpty()) { "Audio samples cannot be empty" }

        val effectiveThreads = SpeechThreadingSupport.resolveThreadCount(nThreads)
        val langId = detectLanguageNative(samples, effectiveThreads)
        return if (langId >= 0) resolveLanguageString(langId) else null
    }
}
