package io.aatricks.llmedge.image.ipc

import android.content.Context
import android.os.IBinder
import android.os.SharedMemory
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.WorkerKilledByMemoryException
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.image.UpscaleRequest
import io.aatricks.llmedge.model.ModelSpec
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import java.io.File
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class IsolatedDiffusionEngineTest {

    private lateinit var context: Context
    private lateinit var scope: LLMEdgeScope
    private lateinit var connectionManager: WorkerConnectionManager
    private lateinit var connection: WorkerConnectionManager.Connection
    private lateinit var worker: IDiffusionWorker
    private lateinit var binder: IBinder
    private lateinit var engine: IsolatedDiffusionEngine

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        scope = LLMEdgeScope(TestScope(), 1)
        connectionManager = mockk(relaxed = true)
        worker = mockk(relaxed = true)
        binder = mockk(relaxed = true)
        connection = WorkerConnectionManager.Connection(worker, binder, 12345)

        coEvery { connectionManager.connect(any()) } returns connection

        mockkObject(WorkerFailureClassifier)
        mockkObject(PixelCodec)

        engine = IsolatedDiffusionEngine(
            context = context,
            edgeScope = scope,
            config = LLMEdgeConfig(),
            connectionManager = connectionManager
        )
    }

    @After
    fun teardown() {
        unmockkAll()
        clearAllMocks()
    }

    @Test
    fun `automatic split request retries sequentially after worker OOM during loading`() = runTest {
        val t5Spec = ModelSpec.LocalFile(File("t5"))
        val clipL = ModelSpec.LocalFile(File("clipL"))
        val clipG = ModelSpec.LocalFile(File("clipG"))

        val params = ImageGenerationRequest(
            prompt = "hello",
            splitDiffusionModel = true,
            t5xxl = t5Spec,
            clipL = clipL,
            clipG = clipG,
        )

        // Mock WorkerFailureClassifier to return WorkerKilledByMemoryException
        every {
            WorkerFailureClassifier.classify(any(), any(), any(), any(), any(), any())
        } returns WorkerKilledByMemoryException()

        // Capture request params without mockk capture slots (which can be flaky)
        val requestSlots = mutableListOf<IpcImageRequest>()
        val callbackSlots = mutableListOf<IDiffusionResultCallback>()

        every {
            worker.generateImage(any(), any())
        } answers {
            val request = firstArg<IpcImageRequest>()
            val callback = secondArg<IDiffusionResultCallback>()
            requestSlots.add(request)
            callbackSlots.add(callback)

            // First call triggers OOM death simulation
            if (requestSlots.size == 1) {
                callback.onPhase(
                    PhaseUpdate(
                        phase = DiffusionPhases.LOADING,
                        backend = "VULKAN",
                        step = 0,
                        totalSteps = 0,
                        uptimeMillis = 0L,
                    ),
                )
                connection.onDeath?.invoke()
            } else {
                // Second call (retry) completes successfully
                val shm = SharedMemory.create("test", 12)
                val mockFrame = IpcFrameBuffer(shm, 1, 1, 1)
                val mockMetrics = IpcGenerationMetrics(
                    totalTimeSeconds = 1.0f,
                    framesPerSecond = 1.0f,
                    timePerStep = 1.0f,
                    peakMemoryUsageMb = 100L,
                    vulkanEnabled = false,
                    frameConversionTimeSeconds = 0.1f,
                    hasRequestMetrics = false,
                    runtimeAcquireMs = 0L,
                    modelLoadMs = 0L,
                    generateMs = 0L,
                    cacheHit = false,
                    backend = "CPU",
                    flashAttentionEnabled = false,
                    easyCacheEnabled = false,
                    width = 1,
                    height = 1,
                    steps = 1
                )
                val mockResult = IpcImageResult(mockFrame, mockMetrics)
                callback.onCompleted(mockResult)
            }
        }

        // Mock PixelCodec.decodeBitmap to avoid real Android Bitmap creation issues in unit tests
        val mockBitmap = mockk<android.graphics.Bitmap>(relaxed = true)
        every { PixelCodec.decodeBitmap(any()) } returns mockBitmap

        val result = engine.generate(params)
        assertEquals(mockBitmap, result)

        // Verify that two requests were sent to the worker
        assertEquals(2, requestSlots.size)
        // Verify first request was automatic.
        assertNull(requestSlots[0].sequential)
        // Verify second request was sequential
        assertEquals(true, requestSlots[1].sequential)
    }

    @Test
    fun `automatic request does not retry after worker OOM during generation`() = runTest {
        val t5Spec = ModelSpec.LocalFile(File("t5"))
        val clipL = ModelSpec.LocalFile(File("clipL"))
        val clipG = ModelSpec.LocalFile(File("clipG"))

        val params = ImageGenerationRequest(
            prompt = "hello",
            splitDiffusionModel = true,
            t5xxl = t5Spec,
            clipL = clipL,
            clipG = clipG,
        )

        every {
            WorkerFailureClassifier.classify(any(), any(), any(), any(), any(), any())
        } returns WorkerKilledByMemoryException()

        val requestSlots = mutableListOf<IpcImageRequest>()
        val callbackSlots = mutableListOf<IDiffusionResultCallback>()

        every {
            worker.generateImage(any(), any())
        } answers {
            val request = firstArg<IpcImageRequest>()
            val callback = secondArg<IDiffusionResultCallback>()
            requestSlots.add(request)
            callbackSlots.add(callback)

            callback.onPhase(
                PhaseUpdate(
                    phase = DiffusionPhases.GENERATING,
                    backend = "VULKAN",
                    step = 0,
                    totalSteps = 0,
                    uptimeMillis = 0L,
                ),
            )
            connection.onDeath?.invoke()
        }

        var exceptionThrown = false
        try {
            engine.generate(params)
        } catch (e: WorkerKilledByMemoryException) {
            exceptionThrown = true
        }

        assertTrue(exceptionThrown)
        assertEquals(1, requestSlots.size)
    }

    @Test
    fun `CPU-session crash implicating the vulkan driver retries once with vulkan quarantined`() = runTest {
        val params = ImageGenerationRequest(prompt = "hello")

        every {
            WorkerFailureClassifier.classify(any(), any(), any(), any(), any(), any())
        } returns io.aatricks.llmedge.core.WorkerCrashedException(
            backend = "CPU",
            exitReason = 5,
            crashSummary = "phase=GENERATING; SIGSEGV | /vendor/lib64/hw/mt6855/vulkan.mtk.so | base.apk!libsdcpp.so",
        )

        val initConfigs = mutableListOf<WorkerInitConfig>()
        coEvery { connectionManager.connect(any()) } answers {
            initConfigs.add(firstArg())
            connection
        }

        val requestSlots = mutableListOf<IpcImageRequest>()
        every {
            worker.generateImage(any(), any())
        } answers {
            val callback = secondArg<IDiffusionResultCallback>()
            requestSlots.add(firstArg())
            if (requestSlots.size == 1) {
                connection.onDeath?.invoke()
            } else {
                val shm = SharedMemory.create("test", 12)
                callback.onCompleted(IpcImageResult(IpcFrameBuffer(shm, 1, 1, 1), null))
            }
        }
        val mockBitmap = mockk<android.graphics.Bitmap>(relaxed = true)
        every { PixelCodec.decodeBitmap(any()) } returns mockBitmap

        val result = engine.generate(params)

        assertEquals(mockBitmap, result)
        assertEquals(2, requestSlots.size)
        assertEquals(false, initConfigs[1].useVulkan)
        assertTrue(
            "expected persisted IMAGE:VULKAN verdict",
            BackendVerdictStore(context).load().contains(
                io.aatricks.llmedge.runtime.ComputeSubsystem.IMAGE to
                    io.aatricks.llmedge.runtime.ComputeBackend.VULKAN,
            ),
        )
    }

    @Test
    fun `CPU-session crash without vulkan driver evidence is not retried`() = runTest {
        val params = ImageGenerationRequest(prompt = "hello")

        every {
            WorkerFailureClassifier.classify(any(), any(), any(), any(), any(), any())
        } returns io.aatricks.llmedge.core.WorkerCrashedException(
            backend = "CPU",
            exitReason = 5,
            // The prior-hang marker names VULKAN without implicating the driver in this crash.
            crashSummary = "phase=GENERATING; gpu-disabled-after-prior-hang(IMAGE:VULKAN)",
        )

        val requestSlots = mutableListOf<IpcImageRequest>()
        every {
            worker.generateImage(any(), any())
        } answers {
            requestSlots.add(firstArg())
            connection.onDeath?.invoke()
        }

        val error = runCatching { engine.generate(params) }.exceptionOrNull()

        assertTrue(error is io.aatricks.llmedge.core.WorkerCrashedException)
        assertEquals(1, requestSlots.size)
        assertTrue(BackendVerdictStore(context).load().isEmpty())
    }

    @Test
    fun `vulkan create-failure marker from the worker becomes a persisted verdict`() = runTest {
        io.aatricks.llmedge.image.diffusion.VulkanCreateFailureMarker.record(context)

        val freshEngine = IsolatedDiffusionEngine(
            context = context,
            edgeScope = scope,
            config = LLMEdgeConfig(),
            connectionManager = connectionManager,
        )

        val verdicts = BackendVerdictStore(context).load()
        assertTrue(
            "expected IMAGE:VULKAN verdict from marker, got $verdicts",
            verdicts.contains(
                io.aatricks.llmedge.runtime.ComputeSubsystem.IMAGE to
                    io.aatricks.llmedge.runtime.ComputeBackend.VULKAN,
            ),
        )
        assertTrue(
            "marker file should be consumed",
            !io.aatricks.llmedge.image.diffusion.VulkanCreateFailureMarker.file(context).exists(),
        )
        freshEngine.close()
    }

    @Test
    fun `forced direct request does not retry after worker OOM during loading`() = runTest {
        val params =
            ImageGenerationRequest(
                prompt = "hello",
                splitDiffusionModel = true,
                textEncoder = ModelSpec.LocalFile(File("encoder")),
                sequential = false,
            )
        every {
            WorkerFailureClassifier.classify(any(), any(), any(), any(), any(), any())
        } returns WorkerKilledByMemoryException()
        val requestSlots = mutableListOf<IpcImageRequest>()

        every {
            worker.generateImage(any(), any())
        } answers {
            val request = firstArg<IpcImageRequest>()
            val callback = secondArg<IDiffusionResultCallback>()
            requestSlots += request
            callback.onPhase(
                PhaseUpdate(
                    phase = DiffusionPhases.LOADING,
                    backend = "VULKAN",
                    step = 0,
                    totalSteps = 0,
                    uptimeMillis = 0L,
                ),
            )
            connection.onDeath?.invoke()
        }

        val error = runCatching { engine.generate(params) }.exceptionOrNull()

        assertTrue(error is WorkerKilledByMemoryException)
        assertEquals(1, requestSlots.size)
        assertEquals(false, requestSlots.single().sequential)
    }

    @Test
    fun `generateStream emits Progress on STEP and Completed on onCompleted`() = runTest {
        val params = ImageGenerationRequest(prompt = "hello")

        val callbackSlots = mutableListOf<IDiffusionResultCallback>()
        every {
            worker.generateImage(any(), any())
        } answers {
            callbackSlots.add(secondArg<IDiffusionResultCallback>())
        }

        val shm = SharedMemory.create("test", 12)
        val mockFrame = IpcFrameBuffer(shm, 1, 1, 1)
        val mockResult = IpcImageResult(mockFrame, null)
        val mockBitmap = mockk<android.graphics.Bitmap>(relaxed = true)
        every { PixelCodec.decodeBitmap(any()) } returns mockBitmap

        val events = mutableListOf<io.aatricks.llmedge.image.GenerationStreamEvent>()
        // The shared `engine` uses a detached TestScope whose scheduler never advances
        // under this runTest; generateStream launches its producer there and would hang.
        // Bind a local engine to this test's scope instead.
        val streamScope = LLMEdgeScope(this, 1)
        val streamEngine = IsolatedDiffusionEngine(
            context = context,
            edgeScope = streamScope,
            config = LLMEdgeConfig(),
            connectionManager = connectionManager,
        )
        val job = launch {
            streamEngine.generateStream(params).collect {
                events.add(it)
            }
        }

        while (callbackSlots.isEmpty()) {
            kotlinx.coroutines.yield()
        }
        val callback = callbackSlots.first()

        callback.onPhase(
            PhaseUpdate(
                phase = DiffusionPhases.STEP,
                backend = null,
                step = 5,
                totalSteps = 20,
                uptimeMillis = 100L
            )
        )
        callback.onCompleted(mockResult)
        job.join()
        streamScope.close()

        assertEquals(2, events.size)
        val progress = events[0] as io.aatricks.llmedge.image.GenerationStreamEvent.Progress
        assertEquals("Sampling", progress.update.message)
        assertEquals(5, progress.update.current)
        assertEquals(20, progress.update.total)

        val completed = events[1] as io.aatricks.llmedge.image.GenerationStreamEvent.Completed
        assertEquals(1, completed.frames.size)
        assertEquals(mockBitmap, completed.frames[0])
    }

    @Test
    fun `generateStream emits no Progress for non-STEP phase updates`() = runTest {
        val params = ImageGenerationRequest(prompt = "hello")

        val callbackSlots = mutableListOf<IDiffusionResultCallback>()
        every {
            worker.generateImage(any(), any())
        } answers {
            callbackSlots.add(secondArg<IDiffusionResultCallback>())
        }

        val shm = SharedMemory.create("test", 12)
        val mockFrame = IpcFrameBuffer(shm, 1, 1, 1)
        val mockResult = IpcImageResult(mockFrame, null)
        val mockBitmap = mockk<android.graphics.Bitmap>(relaxed = true)
        every { PixelCodec.decodeBitmap(any()) } returns mockBitmap

        val events = mutableListOf<io.aatricks.llmedge.image.GenerationStreamEvent>()
        // The shared `engine` uses a detached TestScope whose scheduler never advances
        // under this runTest; generateStream launches its producer there and would hang.
        // Bind a local engine to this test's scope instead.
        val streamScope = LLMEdgeScope(this, 1)
        val streamEngine = IsolatedDiffusionEngine(
            context = context,
            edgeScope = streamScope,
            config = LLMEdgeConfig(),
            connectionManager = connectionManager,
        )
        val job = launch {
            streamEngine.generateStream(params).collect {
                events.add(it)
            }
        }

        while (callbackSlots.isEmpty()) {
            kotlinx.coroutines.yield()
        }
        val callback = callbackSlots.first()

        callback.onPhase(
            PhaseUpdate(
                phase = DiffusionPhases.LOADING,
                backend = "CPU",
                step = 0,
                totalSteps = 0,
                uptimeMillis = 100L
            )
        )
        callback.onCompleted(mockResult)
        job.join()
        streamScope.close()

        assertEquals(1, events.size)
        assertTrue(events[0] is io.aatricks.llmedge.image.GenerationStreamEvent.Completed)
    }

    @Test
    fun `generateStream drives watchdog with step updates`() = runTest {
        io.mockk.mockkConstructor(GenerationWatchdog::class)
        every { anyConstructed<GenerationWatchdog>().onStep(any(), any()) } returns Unit
        every { anyConstructed<GenerationWatchdog>().onPhase(any(), any()) } returns Unit

        val params = ImageGenerationRequest(prompt = "hello")

        val callbackSlots = mutableListOf<IDiffusionResultCallback>()
        every {
            worker.generateImage(any(), any())
        } answers {
            callbackSlots.add(secondArg<IDiffusionResultCallback>())
        }

        val shm = SharedMemory.create("test", 12)
        val mockFrame = IpcFrameBuffer(shm, 1, 1, 1)
        val mockResult = IpcImageResult(mockFrame, null)
        val mockBitmap = mockk<android.graphics.Bitmap>(relaxed = true)
        every { PixelCodec.decodeBitmap(any()) } returns mockBitmap

        // The shared `engine` uses a detached TestScope whose scheduler never advances
        // under this runTest; generateStream launches its producer there and would hang.
        // Bind a local engine to this test's scope instead.
        val streamScope = LLMEdgeScope(this, 1)
        val streamEngine = IsolatedDiffusionEngine(
            context = context,
            edgeScope = streamScope,
            config = LLMEdgeConfig(),
            connectionManager = connectionManager,
        )
        val job = launch {
            streamEngine.generateStream(params).collect {}
        }

        while (callbackSlots.isEmpty()) {
            kotlinx.coroutines.yield()
        }
        val callback = callbackSlots.first()

        callback.onPhase(
            PhaseUpdate(
                phase = DiffusionPhases.STEP,
                backend = null,
                step = 7,
                totalSteps = 15,
                uptimeMillis = 100L
            )
        )
        callback.onCompleted(mockResult)
        job.join()
        streamScope.close()

        io.mockk.verify {
            anyConstructed<GenerationWatchdog>().onStep(7, 15)
        }
    }

    @Test
    fun `upscale completes successfully`() = runTest {
        val model = ModelSpec.LocalFile(File("esrgan.bin"))
        // Real bitmap: mocking the final Bitmap class trips Robolectric's shadow retransformation.
        val inputBitmap = android.graphics.Bitmap.createBitmap(64, 64, android.graphics.Bitmap.Config.ARGB_8888)
        val request = UpscaleRequest(input = inputBitmap, model = model, useVulkan = true)
        every { PixelCodec.encodeBitmap(any(), any()) } answers {
            IpcFrameBuffer(SharedMemory.create("upscale-input", 12), 1, 1, 1)
        }

        val requestSlots = mutableListOf<IpcUpscaleRequest>()
        val callbackSlots = mutableListOf<IDiffusionResultCallback>()

        every {
            worker.upscaleImage(any(), any())
        } answers {
            requestSlots.add(firstArg<IpcUpscaleRequest>())
            val callback = secondArg<IDiffusionResultCallback>()
            callbackSlots.add(callback)

            callback.onPhase(
                PhaseUpdate(
                    phase = DiffusionPhases.STEP,
                    backend = null,
                    step = 2,
                    totalSteps = 4,
                    uptimeMillis = 0L,
                )
            )

            val shm = SharedMemory.create("test", 12)
            val mockFrame = IpcFrameBuffer(shm, 1, 1, 1)
            val mockResult = IpcImageResult(mockFrame, null)
            callback.onCompleted(mockResult)
        }

        val mockBitmap = mockk<android.graphics.Bitmap>(relaxed = true)
        every { PixelCodec.decodeBitmap(any()) } returns mockBitmap

        val progressList = mutableListOf<Pair<Int, Int>>()
        val result = engine.upscale(request) { current, total ->
            progressList.add(current to total)
        }
        assertEquals(mockBitmap, result)
        assertEquals(1, requestSlots.size)
        assertTrue(requestSlots[0].useVulkan)
        assertEquals(listOf(2 to 4), progressList)
    }

    @Test
    fun `upscale failure maps to InferenceFailedException`() = runTest {
        val model = ModelSpec.LocalFile(File("esrgan.bin"))
        // Real bitmap: mocking the final Bitmap class trips Robolectric's shadow retransformation.
        val inputBitmap = android.graphics.Bitmap.createBitmap(64, 64, android.graphics.Bitmap.Config.ARGB_8888)
        val request = UpscaleRequest(input = inputBitmap, model = model, useVulkan = true)
        every { PixelCodec.encodeBitmap(any(), any()) } answers {
            IpcFrameBuffer(SharedMemory.create("upscale-input", 12), 1, 1, 1)
        }

        every {
            worker.upscaleImage(any(), any())
        } answers {
            val callback = secondArg<IDiffusionResultCallback>()
            callback.onFailed(
                IpcFailure(
                    code = IpcFailure.CODE_GENERIC,
                    exceptionClass = "java.lang.RuntimeException",
                    message = "Simulated upscale failure",
                    backend = "CPU"
                )
            )
        }

        var threw = false
        try {
            engine.upscale(request)
        } catch (e: io.aatricks.llmedge.core.InferenceFailedException) {
            threw = true
            assertTrue(e.message?.contains("Simulated upscale failure") == true)
        }
        assertTrue(threw)
    }

    @Test
    fun `upscale retry occurs when worker dies`() = runTest {
        val model = ModelSpec.LocalFile(File("esrgan.bin"))
        // Real bitmap: mocking the final Bitmap class trips Robolectric's shadow retransformation.
        val inputBitmap = android.graphics.Bitmap.createBitmap(64, 64, android.graphics.Bitmap.Config.ARGB_8888)
        val request = UpscaleRequest(input = inputBitmap, model = model, useVulkan = true)
        every { PixelCodec.encodeBitmap(any(), any()) } answers {
            IpcFrameBuffer(SharedMemory.create("upscale-input", 12), 1, 1, 1)
        }

        // Mock WorkerFailureClassifier to return WorkerCrashedException (implicating Vulkan)
        every {
            WorkerFailureClassifier.classify(any(), any(), any(), any(), any(), any())
        } returns io.aatricks.llmedge.core.WorkerCrashedException(
            backend = "VULKAN",
            exitReason = 5,
            crashSummary = "phase=GENERATING; SIGSEGV | /vendor/lib64/hw/mt6855/vulkan.mtk.so | base.apk!libsdcpp.so",
        )

        val requestSlots = mutableListOf<IpcUpscaleRequest>()
        every {
            worker.upscaleImage(any(), any())
        } answers {
            val callback = secondArg<IDiffusionResultCallback>()
            requestSlots.add(firstArg())
            if (requestSlots.size == 1) {
                // Simulate worker death
                connection.onDeath?.invoke()
            } else {
                val shm = SharedMemory.create("test", 12)
                callback.onCompleted(IpcImageResult(IpcFrameBuffer(shm, 1, 1, 1), null))
            }
        }

        val mockBitmap = mockk<android.graphics.Bitmap>(relaxed = true)
        every { PixelCodec.decodeBitmap(any()) } returns mockBitmap

        val result = engine.upscale(request)
        assertEquals(mockBitmap, result)
        assertEquals(2, requestSlots.size)
        assertTrue(requestSlots[0].useVulkan)
        assertFalse(requestSlots[1].useVulkan)
    }
}
