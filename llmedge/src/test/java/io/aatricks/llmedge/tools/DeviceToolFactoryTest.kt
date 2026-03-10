package io.aatricks.llmedge.tools

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class DeviceToolFactoryTest {
    @Test
    fun `time and device tools return useful text`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val factory = DeviceToolFactory(context)

        val timeText = factory.createGetTimeTool().execute(emptyMap())
        val deviceText = factory.createGetDeviceInfoTool().execute(emptyMap())

        assertTrue(timeText.contains("Current Date and Time:"))
        assertTrue(deviceText.contains("Manufacturer:"))
        assertTrue(deviceText.contains("Model:"))
    }

    @Test
    fun `open browser tool returns helpful error without url`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val factory = DeviceToolFactory(context)

        val result = factory.createOpenBrowserTool().execute(emptyMap())

        assertTrue(result.contains("No URL provided"))
    }
}