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
    fun `text availability under quarantine skips SmolLM isVulkanBackendAvailable`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        
        io.mockk.mockkObject(io.aatricks.llmedge.text.runtime.SmolLM)
        io.mockk.every { io.aatricks.llmedge.text.runtime.SmolLM.isOpenClAvailable() } returns true
        io.mockk.every { io.aatricks.llmedge.text.runtime.SmolLM.isVulkanBackendAvailable() } returns true
        
        // Setup quarantine
        io.aatricks.llmedge.image.ipc.BackendVerdictStore(context).recordHang(
            io.aatricks.llmedge.runtime.ComputeSubsystem.IMAGE,
            io.aatricks.llmedge.runtime.ComputeBackend.VULKAN
        )
        
        val availability = LLMEdge.getTextBackendAvailability(context)
        org.junit.Assert.assertFalse(availability.vulkanAvailable)
        org.junit.Assert.assertTrue(availability.openClAvailable)
        
        io.mockk.verify(exactly = 0) {
            io.aatricks.llmedge.text.runtime.SmolLM.isVulkanBackendAvailable()
        }
        
        io.mockk.unmockkObject(io.aatricks.llmedge.text.runtime.SmolLM)
    }
}
