package io.aatricks.llmedge.image.ipc

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class BackendVerdictStoreTest {
    private lateinit var context: Context
    private lateinit var store: BackendVerdictStore

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        store = BackendVerdictStore(context)
        store.reset()
    }

    @Test
    fun `record and load round trip`() {
        store.recordHang(ComputeSubsystem.IMAGE, ComputeBackend.VULKAN)
        store.recordHang(ComputeSubsystem.VIDEO, ComputeBackend.OPENCL)
        val loaded = store.load()
        assertEquals(
            setOf(
                ComputeSubsystem.IMAGE to ComputeBackend.VULKAN,
                ComputeSubsystem.VIDEO to ComputeBackend.OPENCL,
            ),
            loaded.toSet(),
        )
    }

    @Test
    fun `cpu verdicts are never recorded`() {
        store.recordHang(ComputeSubsystem.IMAGE, ComputeBackend.CPU)
        assertTrue(store.load().isEmpty())
    }

    @Test
    fun `fingerprint mismatch clears all verdicts`() {
        store.recordHang(ComputeSubsystem.IMAGE, ComputeBackend.VULKAN)
        // Simulate an OS/driver update by rewriting the stored fingerprint.
        context.getSharedPreferences("llmedge_backend_verdicts", Context.MODE_PRIVATE)
            .edit()
            .putString("fingerprint", "some/other/build:fingerprint")
            .commit()
        assertTrue(store.load().isEmpty())
        // And the store was wiped, not just filtered.
        assertTrue(
            context.getSharedPreferences("llmedge_backend_verdicts", Context.MODE_PRIVATE)
                .all.isEmpty(),
        )
    }

    @Test
    fun `reset clears everything`() {
        store.recordHang(ComputeSubsystem.IMAGE, ComputeBackend.VULKAN)
        store.reset()
        assertTrue(store.load().isEmpty())
    }

    @Test
    fun `recordImageProbe and loadImageProbe round trip`() {
        val availability = io.aatricks.llmedge.ComputeBackendAvailability(
            openClAvailable = true,
            vulkanAvailable = true,
            vulkanDeviceInfo = io.aatricks.llmedge.VulkanDeviceInfo(
                deviceCount = 2,
                freeMemoryMB = 1000L,
                totalMemoryMB = 2000L,
                deviceIndex = 0
            )
        )
        store.recordImageProbe(availability)
        val loaded = store.loadImageProbe()
        assertEquals(availability, loaded)
    }

    @Test
    fun `loadImageProbe returns null on fingerprint mismatch`() {
        val availability = io.aatricks.llmedge.ComputeBackendAvailability(
            openClAvailable = true,
            vulkanAvailable = true,
            vulkanDeviceInfo = null
        )
        store.recordImageProbe(availability)
        
        context.getSharedPreferences("llmedge_backend_verdicts", Context.MODE_PRIVATE)
            .edit()
            .putString("fingerprint", "some/other/build:fingerprint")
            .commit()
            
        org.junit.Assert.assertNull(store.loadImageProbe())
    }
}
