package io.aatricks.llmedge

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class LLMEdgeTest {
    @Test
    fun `create exposes domain clients`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val edge = LLMEdge.create(context, CoroutineScope(SupervisorJob()))

        try {
            assertNotNull(edge.models)
            assertNotNull(edge.text)
            assertNotNull(edge.speech)
            assertNotNull(edge.image)
            assertNotNull(edge.vision)
            assertNotNull(edge.rag)
        } finally {
            edge.close()
        }
    }
    @Test
    fun `no-arg getImageBackendAvailability with no cache returns all-false and never touches StableDiffusion`() {
        // Robolectric shares the sandbox across same-config classes; clear any leaked memo.
        io.aatricks.llmedge.image.ipc.WorkerBackendProber::class.java.getDeclaredField("cached")
            .apply { isAccessible = true }
            .set(io.aatricks.llmedge.image.ipc.WorkerBackendProber, null)
        io.mockk.mockkObject(io.aatricks.llmedge.image.diffusion.StableDiffusion)
        
        val availability = LLMEdge.getImageBackendAvailability()
        org.junit.Assert.assertFalse(availability.vulkanAvailable)
        org.junit.Assert.assertFalse(availability.openClAvailable)
        
        io.mockk.verify(exactly = 0) {
            io.aatricks.llmedge.image.diffusion.StableDiffusion.isOpenClAvailable()
        }
        io.mockk.verify(exactly = 0) {
            io.aatricks.llmedge.image.diffusion.StableDiffusion.getVulkanDeviceCount()
        }
        io.mockk.unmockkObject(io.aatricks.llmedge.image.diffusion.StableDiffusion)
    }

    @Test
    fun `text availability context overload derives from worker probe without touching SmolLM`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val cachedField =
            io.aatricks.llmedge.image.ipc.WorkerBackendProber::class.java.getDeclaredField("cached")
        cachedField.isAccessible = true
        cachedField.set(io.aatricks.llmedge.image.ipc.WorkerBackendProber, null)
        io.aatricks.llmedge.image.ipc.BackendVerdictStore(context).reset()

        io.mockk.mockkObject(io.aatricks.llmedge.text.runtime.SmolLM)

        // Never probed: conservative all-false.
        val before = LLMEdge.getTextBackendAvailability(context)
        org.junit.Assert.assertFalse(before.vulkanAvailable)
        org.junit.Assert.assertFalse(before.openClAvailable)

        // A persisted worker probe drives the answer.
        io.aatricks.llmedge.image.ipc.BackendVerdictStore(context).recordImageProbe(
            ComputeBackendAvailability(false, true, VulkanDeviceInfo(1, 100, 200, 0)),
        )
        val after = LLMEdge.getTextBackendAvailability(context)
        org.junit.Assert.assertTrue(after.vulkanAvailable)

        io.mockk.verify(exactly = 0) {
            io.aatricks.llmedge.text.runtime.SmolLM.isVulkanBackendAvailable()
        }
        io.mockk.verify(exactly = 0) {
            io.aatricks.llmedge.text.runtime.SmolLM.isOpenClAvailable()
        }

        io.mockk.unmockkObject(io.aatricks.llmedge.text.runtime.SmolLM)
    }
}
