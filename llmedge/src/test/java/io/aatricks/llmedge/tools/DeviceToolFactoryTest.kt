package io.aatricks.llmedge.tools

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.serialization.json.buildJsonObject
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class DeviceToolFactoryTest {
    @Test
    fun `default tools include read only tools and optional action tool`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val factory = DeviceToolFactory(context)

        val readOnlyTools = factory.createDefaultTools(includeActions = false)
        val allTools = factory.createDefaultTools(includeActions = true)

        assertEquals(3, readOnlyTools.size)
        assertEquals(4, allTools.size)
        assertTrue(allTools.any { it.name == "open_browser" && it.kind == ToolKind.ACTION })
    }

    @Test
    fun `time and device tools return structured results`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val factory = DeviceToolFactory(context)

        val timeResult = factory.createGetTimeTool().handler(buildJsonObject { })
        val deviceResult = factory.createGetDeviceInfoTool().handler(buildJsonObject { })

        assertTrue(timeResult.text.contains("Current Date and Time:"))
        assertTrue("timestamp" in timeResult.data)
        assertTrue(deviceResult.text.contains("Manufacturer:"))
        assertTrue("model" in deviceResult.data)
    }

    @Test
    fun `open browser tool returns helpful error without url`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val factory = DeviceToolFactory(context)

        val result = factory.createOpenBrowserTool().handler(buildJsonObject { })

        assertTrue(result.isError)
        assertTrue(result.text.contains("No URL provided"))
    }
}
