package io.aatricks.llmedge.speech.stt.internal

import io.aatricks.llmedge.speech.stt.Whisper
import kotlin.math.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

internal class WhisperStreamingState(
    private val whisper: Whisper,
    private val params: Whisper.StreamingParams,
) {
    private val mutex = Mutex()
    private var audioBuffer = FloatArray(0)
    private var audioBufferSize = 0
    private var previousAudio = FloatArray(0)
    private var previousAudioSize = 0

    @Volatile
    private var isRunning = false

    @Volatile
    private var isPaused = false

    private var segmentIndex = 0
    private var totalProcessedMs: Long = 0

    private val samplesPerMs = Whisper.SAMPLE_RATE / 1000
    private val samplesStep = params.stepMs * samplesPerMs
    private val samplesLength = params.lengthMs * samplesPerMs
    private val samplesKeep = min(params.keepMs, params.stepMs) * samplesPerMs

    suspend fun feedAudio(samples: FloatArray) =
        mutex.withLock {
            if (isRunning && !isPaused) {
                val needed = audioBufferSize + samples.size
                if (needed > audioBuffer.size) {
                    audioBuffer = audioBuffer.copyOf(maxOf(needed, audioBuffer.size * 2, 4096))
                }
                System.arraycopy(samples, 0, audioBuffer, audioBufferSize, samples.size)
                audioBufferSize += samples.size
            }
        }

    fun getBufferedAudioMs(): Int = audioBufferSize / samplesPerMs

    fun hasEnoughAudio(): Boolean = audioBufferSize >= samplesStep

    suspend fun clearBuffer() =
        mutex.withLock {
            audioBufferSize = 0
            previousAudioSize = 0
            segmentIndex = 0
            totalProcessedMs = 0
        }

    fun pause() {
        isPaused = true
    }

    fun resume() {
        isPaused = false
    }

    fun stop() {
        isRunning = false
    }

    fun start(): Flow<Whisper.TranscriptionSegment> =
        flow {
            isRunning = true
            isPaused = false

            while (isRunning) {
                val currentBufferSize = mutex.withLock { audioBufferSize }
                if (currentBufferSize < samplesStep) {
                    delay(50)
                    continue
                }

                val segments = processNextChunk()
                for (segment in segments) {
                    emit(segment)
                }
            }
        }.flowOn(Dispatchers.IO)

    suspend fun processNextChunk(): List<Whisper.TranscriptionSegment> =
        mutex.withLock {
            if (audioBufferSize < samplesStep) {
                return@withLock emptyList()
            }

            val newSamplesCount = samplesStep
            val removeCount = min(samplesStep, audioBufferSize)
            val takeFromPrevious = min(previousAudioSize, samplesLength - newSamplesCount)
            val windowSamples = FloatArray(takeFromPrevious + newSamplesCount)

            if (takeFromPrevious > 0) {
                System.arraycopy(previousAudio, previousAudioSize - takeFromPrevious, windowSamples, 0, takeFromPrevious)
            }
            System.arraycopy(audioBuffer, 0, windowSamples, takeFromPrevious, newSamplesCount)

            if (removeCount < audioBufferSize) {
                System.arraycopy(audioBuffer, removeCount, audioBuffer, 0, audioBufferSize - removeCount)
            }
            audioBufferSize -= removeCount

            val keepSamples = min(windowSamples.size, samplesKeep + samplesLength)
            val startIdx = windowSamples.size - keepSamples
            if (keepSamples > previousAudio.size) {
                previousAudio = FloatArray(keepSamples)
            }
            System.arraycopy(windowSamples, startIdx, previousAudio, 0, keepSamples)
            previousAudioSize = keepSamples

            if (params.useVad && !hasVoiceActivity(windowSamples)) {
                return@withLock emptyList()
            }

            val transcribeParams =
                Whisper.TranscribeParams(
                    nThreads = params.nThreads,
                    translate = params.translate,
                    language = params.language,
                    temperature = 0.0f,
                    beamSize = 1,
                )

            val segments = whisper.transcribe(windowSamples, transcribeParams)
            val timeOffsetCentiseconds = (totalProcessedMs / 10).toInt()
            val adjustedSegments =
                segments.map { segment ->
                    Whisper.TranscriptionSegment(
                        index = segmentIndex++,
                        startTime = segment.startTime + timeOffsetCentiseconds,
                        endTime = segment.endTime + timeOffsetCentiseconds,
                        text = segment.text,
                    )
                }

            totalProcessedMs += params.stepMs
            adjustedSegments
        }

    private fun hasVoiceActivity(samples: FloatArray): Boolean {
        if (samples.isEmpty()) return false

        var sumSquares = 0.0
        for (sample in samples) {
            sumSquares += sample * sample
        }
        val rms = kotlin.math.sqrt(sumSquares / samples.size)
        val energyThreshold = params.vadThreshold * 0.02f
        return rms > energyThreshold
    }
}
