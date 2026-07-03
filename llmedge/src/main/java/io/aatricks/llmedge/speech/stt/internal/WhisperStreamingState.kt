package io.aatricks.llmedge.speech.stt.internal

import io.aatricks.llmedge.speech.stt.Whisper
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

internal class WhisperStreamingState(
    private val transcribe: (FloatArray, Whisper.TranscribeParams) -> List<Whisper.TranscriptionSegment>,
    private val params: Whisper.StreamingParams,
) {
    constructor(whisper: Whisper, params: Whisper.StreamingParams) : this(whisper::transcribe, params)

    private val mutex = Mutex()
    private var audioBuffer = FloatArray(0)

    @Volatile
    private var audioBufferSize = 0
    private var previousAudio = FloatArray(0)
    private var previousAudioSize = 0

    @Volatile
    private var isRunning = false

    @Volatile
    private var isStopped = false

    @Volatile
    private var isPaused = false

    private var segmentIndex = 0

    /** Absolute stream time (ms) of audio consumed so far, including dropped/skipped audio. */
    private var totalProcessedMs: Long = 0

    /** Absolute end (centiseconds) of the last emitted segment, for overlap dedup. */
    private var lastEmittedEndCs: Long = 0

    private val samplesPerMs = Whisper.SAMPLE_RATE / 1000
    private val samplesStep = params.stepMs * samplesPerMs
    private val samplesLength = params.lengthMs * samplesPerMs
    private val samplesKeep = min(params.keepMs, params.stepMs) * samplesPerMs

    init {
        require(params.stepMs in 1..params.lengthMs) {
            "stepMs (${params.stepMs}) must be in [1, lengthMs (${params.lengthMs})]"
        }
        require(params.lengthMs <= MAX_WINDOW_MS) {
            "lengthMs (${params.lengthMs}) must not exceed $MAX_WINDOW_MS (whisper window)"
        }
        require(params.keepMs >= 0) { "keepMs must be >= 0" }
    }

    suspend fun feedAudio(samples: FloatArray) =
        mutex.withLock {
            if (isStopped || isPaused) {
                return@withLock
            }
            val needed = audioBufferSize + samples.size
            if (needed > audioBuffer.size) {
                audioBuffer = audioBuffer.copyOf(maxOf(needed, audioBuffer.size * 2, 4096))
            }
            System.arraycopy(samples, 0, audioBuffer, audioBufferSize, samples.size)
            audioBufferSize += samples.size

            // Transcription slower than real time must not grow the backlog without
            // bound: drop the oldest audio and advance the clock so timestamps stay
            // aligned with real time across the gap.
            val maxBuffered = samplesLength * MAX_BACKLOG_WINDOWS
            if (audioBufferSize > maxBuffered) {
                val drop = audioBufferSize - maxBuffered
                System.arraycopy(audioBuffer, drop, audioBuffer, 0, maxBuffered)
                audioBufferSize = maxBuffered
                totalProcessedMs += drop / samplesPerMs
                previousAudioSize = 0
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
            lastEmittedEndCs = 0
        }

    fun pause() {
        isPaused = true
    }

    fun resume() {
        isPaused = false
    }

    fun stop() {
        isStopped = true
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

            // Consume up to a full window of backlog per pass (at least one step) so
            // a slow device catches up instead of queueing further behind.
            val newSamplesCount =
                min(audioBufferSize, max(samplesStep, samplesLength - samplesKeep))
            // Only keepMs of already-transcribed audio is prepended, purely as
            // acoustic context — not a full re-transcription of the prior window.
            val takeFromPrevious = min(previousAudioSize, samplesKeep)
            val windowStartMs = totalProcessedMs - takeFromPrevious / samplesPerMs

            val windowSamples = FloatArray(takeFromPrevious + newSamplesCount)
            if (takeFromPrevious > 0) {
                System.arraycopy(
                    previousAudio,
                    previousAudioSize - takeFromPrevious,
                    windowSamples,
                    0,
                    takeFromPrevious,
                )
            }
            System.arraycopy(audioBuffer, 0, windowSamples, takeFromPrevious, newSamplesCount)

            if (newSamplesCount < audioBufferSize) {
                System.arraycopy(
                    audioBuffer,
                    newSamplesCount,
                    audioBuffer,
                    0,
                    audioBufferSize - newSamplesCount,
                )
            }
            audioBufferSize -= newSamplesCount

            val keepSamples = min(windowSamples.size, samplesKeep)
            if (keepSamples > previousAudio.size) {
                previousAudio = FloatArray(keepSamples)
            }
            System.arraycopy(
                windowSamples,
                windowSamples.size - keepSamples,
                previousAudio,
                0,
                keepSamples,
            )
            previousAudioSize = keepSamples

            // The clock advances for every consumed sample — including windows VAD
            // skips below — or the emitted timeline drifts behind real time.
            totalProcessedMs += newSamplesCount.toLong() / samplesPerMs

            if (params.useVad && !hasVoiceActivity(windowSamples, takeFromPrevious)) {
                return@withLock emptyList()
            }

            val transcribeParams =
                Whisper.TranscribeParams(
                    nThreads = params.nThreads,
                    translate = params.translate,
                    language = params.language,
                    temperature = 0.0f,
                    beamSize = 1,
                    // Windows are independent utterances; shrinking the encoder to
                    // the window avoids a full 30s-padded encode per step.
                    noContext = true,
                    audioCtx = streamingAudioCtx(windowSamples.size),
                )

            val segments = transcribe(windowSamples, transcribeParams)
            val offsetCs = windowStartMs / 10
            val emitted = ArrayList<Whisper.TranscriptionSegment>(segments.size)
            for (segment in segments) {
                val startCs = segment.startTime + offsetCs
                val endCs = segment.endTime + offsetCs
                // Segments that end inside the keepMs context region were already
                // emitted by the previous window.
                if (endCs <= lastEmittedEndCs) {
                    continue
                }
                lastEmittedEndCs = endCs
                emitted.add(
                    Whisper.TranscriptionSegment(
                        index = segmentIndex++,
                        startTime = max(startCs, 0L),
                        endTime = endCs,
                        text = segment.text,
                    ),
                )
            }
            emitted
        }

    /**
     * Encoder frames for the actual window (50 frames/second) plus headroom,
     * bounded to whisper's [128, 1500] usable range.
     */
    private fun streamingAudioCtx(windowSampleCount: Int): Int {
        val frames = windowSampleCount / (Whisper.SAMPLE_RATE / 50)
        return (frames + AUDIO_CTX_MARGIN).coerceIn(MIN_AUDIO_CTX, MAX_AUDIO_CTX)
    }

    /**
     * RMS energy over the NEW samples only (the context prefix is old speech and
     * would mask silence), behind a one-pole high-pass at [Whisper.StreamingParams.vadFreqThreshold]
     * so DC offset and low-frequency hum don't register as voice.
     */
    private fun hasVoiceActivity(samples: FloatArray, newStart: Int): Boolean {
        if (newStart >= samples.size) return false

        val dt = 1.0f / Whisper.SAMPLE_RATE
        val rc = 1.0f / (2.0f * Math.PI.toFloat() * params.vadFreqThreshold.coerceAtLeast(1.0f))
        val alpha = rc / (rc + dt)

        var prevX = samples[newStart]
        var prevY = 0.0f
        var sumSquares = 0.0
        var count = 0
        for (i in newStart + 1 until samples.size) {
            val y = alpha * (prevY + samples[i] - prevX)
            prevX = samples[i]
            prevY = y
            sumSquares += y * y
            count++
        }
        if (count == 0) return false

        val rms = sqrt(sumSquares / count)
        return rms > params.vadThreshold * 0.02f
    }

    private companion object {
        const val MAX_WINDOW_MS = 30_000
        const val MAX_BACKLOG_WINDOWS = 4
        const val MIN_AUDIO_CTX = 128
        const val MAX_AUDIO_CTX = 1500
        const val AUDIO_CTX_MARGIN = 64
    }
}
