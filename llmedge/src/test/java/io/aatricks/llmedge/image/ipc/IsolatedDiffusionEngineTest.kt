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
import org.junit.Assert.assertFalse
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
    fun `direct SD3 split request falling back to sequential retry on WorkerKilledByMemoryException`() = runTest {
        val t5Spec = ModelSpec.LocalFile(File("t5"))
        val clipL = ModelSpec.LocalFile(File("clipL"))
        val clipG = ModelSpec.LocalFile(File("clipG"))

        val params = ImageGenerationRequest(
            prompt = "hello",
            splitDiffusionModel = true,
            t5xxl = t5Spec,
            clipL = clipL,
            clipG = clipG,
            sequential = false
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
        // Verify first request was NOT sequential
        assertFalse(requestSlots[0].sequential)
        // Verify second request was sequential
        assertTrue(requestSlots[1].sequential)
    }

    @Test
    fun `does not retry if sequential also dies`() = runTest {
        val t5Spec = ModelSpec.LocalFile(File("t5"))
        val clipL = ModelSpec.LocalFile(File("clipL"))
        val clipG = ModelSpec.LocalFile(File("clipG"))

        val params = ImageGenerationRequest(
            prompt = "hello",
            splitDiffusionModel = true,
            t5xxl = t5Spec,
            clipL = clipL,
            clipG = clipG,
            sequential = false
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

            connection.onDeath?.invoke()
        }

        var exceptionThrown = false
        try {
            engine.generate(params)
        } catch (e: WorkerKilledByMemoryException) {
            exceptionThrown = true
        }

        assertTrue(exceptionThrown)
        // Should only try twice: once for direct, once for sequential, then fail
        assertEquals(2, requestSlots.size)
    }
}
