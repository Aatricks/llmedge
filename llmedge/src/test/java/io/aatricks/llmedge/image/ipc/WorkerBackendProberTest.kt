package io.aatricks.llmedge.image.ipc

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.core.WorkerKilledByMemoryException
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import io.mockk.verify
import io.mockk.coVerify
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
import io.aatricks.llmedge.runtime.ComputeSubsystem
import io.aatricks.llmedge.runtime.ComputeBackend

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class WorkerBackendProberTest {
    private lateinit var context: Context
    private lateinit var store: BackendVerdictStore

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        store = BackendVerdictStore(context)
        store.reset()
        mockkObject(WorkerFailureClassifier)
        
        val cachedField = WorkerBackendProber::class.java.getDeclaredField("cached")
        cachedField.isAccessible = true
        cachedField.set(WorkerBackendProber, null)
        val quarantinedField = WorkerBackendProber::class.java.getDeclaredField("vulkanQuarantined")
        quarantinedField.isAccessible = true
        quarantinedField.set(WorkerBackendProber, false)
    }

    @After
    fun teardown() {
        unmockkAll()
        clearAllMocks()
    }

    @Test
    fun `successful probe maps availability and persists snapshot, second call memoized`() = runTest {
        io.mockk.mockkConstructor(WorkerConnectionManager::class)
        val connection = mockk<WorkerConnectionManager.Connection>(relaxed = true)
        val worker = mockk<IDiffusionWorker>(relaxed = true)
        every { connection.worker } returns worker
        coEvery { anyConstructed<WorkerConnectionManager>().connect(null) } returns connection
        
        every { worker.probeBackends(any()) } returns IpcBackendProbeResult(
            openClAvailable = true,
            vulkanDeviceCount = 1,
            vulkanFreeBytes = 500L * 1024 * 1024,
            vulkanTotalBytes = 1000L * 1024 * 1024
        )

        val result1 = WorkerBackendProber.probe(context)
        assertTrue(result1.openClAvailable)
        assertTrue(result1.vulkanAvailable)
        assertEquals(500L, result1.vulkanDeviceInfo?.freeMemoryMB)
        assertEquals(1, result1.vulkanDeviceInfo?.deviceCount)

        val result2 = WorkerBackendProber.probe(context)
        assertEquals(result1, result2)

        coVerify(exactly = 1) { anyConstructed<WorkerConnectionManager>().connect(null) }
        
        val loaded = store.loadImageProbe()
        assertEquals(result1, loaded)
    }

    @Test
    fun `DeadObjectException from probeBackends persists verdicts, retries once, returns vulkan false`() = runTest {
        io.mockk.mockkConstructor(WorkerConnectionManager::class)
        val connection = mockk<WorkerConnectionManager.Connection>(relaxed = true)
        every { connection.pid } returns 1234
        val worker = mockk<IDiffusionWorker>(relaxed = true)
        every { connection.worker } returns worker
        coEvery { anyConstructed<WorkerConnectionManager>().connect(null) } returns connection

        every {
            WorkerFailureClassifier.classify(any(), any(), any(), any(), any(), any())
        } returns io.aatricks.llmedge.core.WorkerCrashedException(
            backend = "VULKAN",
            exitReason = 5,
            crashSummary = "crash",
        )

        val seeds = mutableListOf<List<String>>()
        every { worker.probeBackends(any()) } answers {
            seeds.add(firstArg())
            if (seeds.size == 1) {
                throw android.os.DeadObjectException()
            } else {
                IpcBackendProbeResult(true, 1, 1024*1024, 1024*1024)
            }
        }

        val result = WorkerBackendProber.probe(context)
        
        assertTrue(result.openClAvailable)
        assertFalse(result.vulkanAvailable)
        assertNull(result.vulkanDeviceInfo)

        coVerify(exactly = 2) { anyConstructed<WorkerConnectionManager>().connect(null) }
        assertEquals(emptyList<String>(), seeds[0])
        assertEquals(listOf("IMAGE:VULKAN"), seeds[1])

        val verdicts = store.load()
        assertTrue(verdicts.contains(ComputeSubsystem.IMAGE to ComputeBackend.VULKAN))
        assertTrue(verdicts.contains(ComputeSubsystem.VIDEO to ComputeBackend.VULKAN))
        
        assertTrue(WorkerBackendProber.isVulkanQuarantined())
    }

    @Test
    fun `timeout via virtual time invokes killWorker and persists verdict`() = runTest {
        io.mockk.mockkConstructor(WorkerConnectionManager::class)
        val connection = mockk<WorkerConnectionManager.Connection>(relaxed = true)
        val worker = mockk<IDiffusionWorker>(relaxed = true)
        every { connection.worker } returns worker
        coEvery { anyConstructed<WorkerConnectionManager>().connect(null) } returns connection
        every { connection.pid } returns 1234

        every {
            WorkerFailureClassifier.classify(any(), any(), any(), any(), any(), any())
        } returns io.aatricks.llmedge.core.GenerationHangException(backend = "VULKAN", phase = "PROBE", stallMs = 10000L)

        every { worker.probeBackends(any()) } answers {
            val exClass = Class.forName("kotlinx.coroutines.TimeoutCancellationException")
            try {
                throw exClass.getDeclaredConstructor(String::class.java).apply { isAccessible = true }.newInstance("Timeout") as Exception
            } catch (e: NoSuchMethodException) {
                throw exClass.getDeclaredConstructor(String::class.java, kotlinx.coroutines.Job::class.java).apply { isAccessible = true }.newInstance("Timeout", null) as Exception
            }
        }

        val result = WorkerBackendProber.probe(context)
        assertFalse(result.openClAvailable)
        assertFalse(result.vulkanAvailable)

        io.mockk.verify(exactly = 2) { anyConstructed<WorkerConnectionManager>().killWorker(connection) }
        
        val verdicts = store.load()
        assertTrue(verdicts.contains(ComputeSubsystem.IMAGE to ComputeBackend.VULKAN))
    }

    @Test
    fun `pre-existing persisted verdict means connect never called when snapshot exists`() = runTest {
        io.mockk.mockkConstructor(WorkerConnectionManager::class)
        
        store.recordHang(ComputeSubsystem.IMAGE, ComputeBackend.VULKAN)
        store.recordImageProbe(io.aatricks.llmedge.ComputeBackendAvailability(true, true, io.aatricks.llmedge.VulkanDeviceInfo(1, 1000, 2000, 0)))

        val result = WorkerBackendProber.probe(context)
        
        assertTrue(result.openClAvailable)
        assertFalse(result.vulkanAvailable)
        assertNull(result.vulkanDeviceInfo)
        assertTrue(WorkerBackendProber.isVulkanQuarantined())

        coVerify(exactly = 0) { anyConstructed<WorkerConnectionManager>().connect(any()) }
    }

    @Test
    fun `persistedOrNull strips vulkan from snapshot when a vulkan verdict exists`() {
        store.recordImageProbe(io.aatricks.llmedge.ComputeBackendAvailability(true, true, io.aatricks.llmedge.VulkanDeviceInfo(1, 1000, 2000, 0)))
        store.recordHang(ComputeSubsystem.IMAGE, ComputeBackend.VULKAN)

        val result = WorkerBackendProber.persistedOrNull(context)

        assertEquals(true, result?.openClAvailable)
        assertEquals(false, result?.vulkanAvailable)
        assertNull(result?.vulkanDeviceInfo)
        assertTrue(WorkerBackendProber.isVulkanQuarantined())
    }

    @Test
    fun `persisted snapshot short-circuit and fingerprint change invalidates`() = runTest {
        io.mockk.mockkConstructor(WorkerConnectionManager::class)
        
        store.recordImageProbe(io.aatricks.llmedge.ComputeBackendAvailability(true, true, io.aatricks.llmedge.VulkanDeviceInfo(1, 1000, 2000, 0)))
        
        val result1 = WorkerBackendProber.probe(context)
        assertTrue(result1.vulkanAvailable)
        coVerify(exactly = 0) { anyConstructed<WorkerConnectionManager>().connect(any()) }
        
        val cachedField = WorkerBackendProber::class.java.getDeclaredField("cached")
        cachedField.isAccessible = true
        cachedField.set(WorkerBackendProber, null)
        
        context.getSharedPreferences("llmedge_backend_verdicts", Context.MODE_PRIVATE)
            .edit()
            .putString("fingerprint", "different")
            .commit()
            
        val connection = mockk<WorkerConnectionManager.Connection>(relaxed = true)
        val worker = mockk<IDiffusionWorker>(relaxed = true)
        every { connection.worker } returns worker
        coEvery { anyConstructed<WorkerConnectionManager>().connect(null) } returns connection
        every { worker.probeBackends(any()) } returns IpcBackendProbeResult(false, 0, 0, 0)
        
        val result2 = WorkerBackendProber.probe(context)
        assertFalse(result2.vulkanAvailable)
        coVerify(exactly = 1) { anyConstructed<WorkerConnectionManager>().connect(null) }
    }
}
