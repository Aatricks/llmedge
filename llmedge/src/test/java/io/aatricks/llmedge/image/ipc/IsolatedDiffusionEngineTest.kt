package io.aatricks.llmedge.image.ipc

import android.content.Context
import android.os.IBinder
import android.os.SharedMemory
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.WorkerKilledByMemoryException
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.model.ModelSpec
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import java.io.File
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
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
}
