package io.aatricks.llmedge

import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import org.junit.Assert.*
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import io.aatricks.llmedge.speech.stt.Whisper
import java.io.File

/**
 * Unit tests for Whisper class - tests that don't require native library.
 * These test the Kotlin API layer and data classes.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class WhisperTest {
    @After
    fun tearDown() {
        Whisper.resetLoadBridgeForTests()
        Whisper.resetBackendAvailabilityForTests()
        BackendRuntimePolicy.resetForTests()
    }

    @Test
    fun `TranscriptionSegment startTimeMs converts correctly`() {
        val segment = Whisper.TranscriptionSegment(
            index = 0,
            startTime = 100L,  // 100 centiseconds = 1000ms
            endTime = 200L,    // 200 centiseconds = 2000ms
            text = "Test"
        )

        assertEquals(1000L, segment.startTimeMs)
        assertEquals(2000L, segment.endTimeMs)
        assertEquals(1000L, segment.durationMs)
    }

    @Test
    fun `TranscriptionSegment toSrtEntry formats correctly`() {
        val segment = Whisper.TranscriptionSegment(
            index = 0,
            startTime = 0L,
            endTime = 150L,  // 1500ms
            text = "Hello, world!"
        )

        val srt = segment.toSrtEntry()
        assertTrue(srt.contains("1"))  // Index + 1
        assertTrue(srt.contains("-->"))
        assertTrue(srt.contains("Hello, world!"))
        assertTrue(srt.contains(","))  // SRT uses comma for ms separator
    }

    @Test
    fun `TranscriptionSegment toVttEntry formats correctly`() {
        val segment = Whisper.TranscriptionSegment(
            index = 0,
            startTime = 0L,
            endTime = 150L,
            text = "Hello, world!"
        )

        val vtt = segment.toVttEntry()
        assertTrue(vtt.contains("-->"))
        assertTrue(vtt.contains("Hello, world!"))
        assertTrue(vtt.contains("."))  // VTT uses period for ms separator
    }

    @Test
    fun `TranscribeParams default values are correct`() {
        val params = Whisper.TranscribeParams()

        assertEquals(0, params.nThreads)
        assertFalse(params.translate)
        assertNull(params.language)
        assertFalse(params.detectLanguage)
        assertFalse(params.tokenTimestamps)
        assertEquals(0, params.maxLen)
        assertFalse(params.splitOnWord)
        assertEquals(0.0f, params.temperature, 0.001f)
        assertEquals(1, params.beamSize)
        assertTrue(params.suppressBlank)
        assertFalse(params.printProgress)
    }

    @Test
    fun `TranscribeParams custom values`() {
        val params = Whisper.TranscribeParams(
            nThreads = 4,
            translate = true,
            language = "en",
            detectLanguage = true,
            tokenTimestamps = true,
            maxLen = 100,
            splitOnWord = true,
            temperature = 0.5f,
            beamSize = 5,
            suppressBlank = false,
            printProgress = true
        )

        assertEquals(4, params.nThreads)
        assertTrue(params.translate)
        assertEquals("en", params.language)
        assertTrue(params.detectLanguage)
        assertTrue(params.tokenTimestamps)
        assertEquals(100, params.maxLen)
        assertTrue(params.splitOnWord)
        assertEquals(0.5f, params.temperature, 0.001f)
        assertEquals(5, params.beamSize)
        assertFalse(params.suppressBlank)
        assertTrue(params.printProgress)
    }

    @Test
    fun `SAMPLE_RATE constant is 16000`() {
        assertEquals(16000, Whisper.SAMPLE_RATE)
    }

    @Test
    fun `CHUNK_SIZE_SECONDS constant is 30`() {
        assertEquals(30, Whisper.CHUNK_SIZE_SECONDS)
    }

    @Test
    fun `TranscriptionSegment with zero times`() {
        val segment = Whisper.TranscriptionSegment(
            index = 0,
            startTime = 0L,
            endTime = 0L,
            text = ""
        )

        assertEquals(0L, segment.startTimeMs)
        assertEquals(0L, segment.endTimeMs)
        assertEquals(0L, segment.durationMs)
    }

    @Test
    fun `TranscriptionSegment with large times`() {
        // 1 hour in centiseconds = 360000
        val segment = Whisper.TranscriptionSegment(
            index = 99,
            startTime = 360000L,
            endTime = 360100L,
            text = "After one hour"
        )

        assertEquals(3600000L, segment.startTimeMs)  // 1 hour in ms
        assertEquals(3601000L, segment.endTimeMs)
        assertEquals(1000L, segment.durationMs)
    }

    @Test
    fun `SRT format hour handling`() {
        // Test a segment at 2 hours, 30 minutes, 45 seconds, 500ms
        // = 2*3600000 + 30*60000 + 45*1000 + 500 = 9045500 ms
        // = 904550 centiseconds
        val segment = Whisper.TranscriptionSegment(
            index = 0,
            startTime = 904550L,
            endTime = 904650L,
            text = "Test"
        )

        val srt = segment.toSrtEntry()
        assertTrue("SRT should contain hour value", srt.contains("02:30:45"))
    }

    @Test
    fun `load prefers OpenCL and falls back to Vulkan after OpenCL failure`() {
        val modelFile = File.createTempFile("whisper-load", ".bin").apply {
            writeText("test")
            deleteOnExit()
        }
        val attemptedBackends = mutableListOf<ComputeBackend>()

        Whisper.overrideBackendAvailabilityForTests(openClAvailable = true, vulkanAvailable = true)
        Whisper.overrideLoadBridgeForTests {
            object : Whisper.Companion.LoadBridge {
                override fun create(
                    modelPath: String,
                    backend: ComputeBackend,
                    flashAttn: Boolean,
                    gpuDevice: Int,
                ): Long {
                    attemptedBackends += backend
                    if (backend == ComputeBackend.OPENCL) {
                        throw RuntimeException("OpenCL init failed")
                    }
                    return 7L
                }
            }
        }

        val whisper = Whisper.load(modelFile.absolutePath, useGpu = true)

        assertEquals(listOf(ComputeBackend.OPENCL, ComputeBackend.VULKAN), attemptedBackends)
        assertEquals(ComputeBackend.VULKAN, whisper.activeBackend)
        assertTrue(BackendRuntimePolicy.isBlacklisted(ComputeSubsystem.WHISPER, ComputeBackend.OPENCL))
    }

    @Test
    fun `load skips blacklisted OpenCL backend on later attempts`() {
        val modelFile = File.createTempFile("whisper-blacklist", ".bin").apply {
            writeText("test")
            deleteOnExit()
        }
        val attemptedBackends = mutableListOf<ComputeBackend>()

        BackendRuntimePolicy.blacklist(ComputeSubsystem.WHISPER, ComputeBackend.OPENCL)
        Whisper.overrideBackendAvailabilityForTests(openClAvailable = true, vulkanAvailable = true)
        Whisper.overrideLoadBridgeForTests {
            object : Whisper.Companion.LoadBridge {
                override fun create(
                    modelPath: String,
                    backend: ComputeBackend,
                    flashAttn: Boolean,
                    gpuDevice: Int,
                ): Long {
                    attemptedBackends += backend
                    return 9L
                }
            }
        }

        val whisper = Whisper.load(modelFile.absolutePath, useGpu = true)

        assertEquals(listOf(ComputeBackend.VULKAN), attemptedBackends)
        assertEquals(ComputeBackend.VULKAN, whisper.activeBackend)
    }
}
