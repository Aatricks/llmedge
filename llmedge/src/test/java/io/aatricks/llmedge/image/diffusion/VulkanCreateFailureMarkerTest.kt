package io.aatricks.llmedge.image.diffusion

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VulkanCreateFailureMarkerTest {
    private val context: Context = ApplicationProvider.getApplicationContext()

    @Test
    fun `record then consume returns true once and deletes the marker`() {
        VulkanCreateFailureMarker.record(context)
        assertTrue(VulkanCreateFailureMarker.consume(context))
        assertFalse(VulkanCreateFailureMarker.file(context).exists())
        assertFalse(VulkanCreateFailureMarker.consume(context))
    }

    @Test
    fun `marker from a previous OS build is discarded`() {
        VulkanCreateFailureMarker.file(context).writeText("some-old-fingerprint")
        assertFalse(VulkanCreateFailureMarker.consume(context))
        assertFalse(VulkanCreateFailureMarker.file(context).exists())
    }
}
