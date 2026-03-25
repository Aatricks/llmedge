package io.aatricks.llmedge.tools

import android.app.Application
import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
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
    fun `battery tool returns integer percent with source metadata`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val factory = DeviceToolFactory(context)

        val result = factory.createGetBatteryStatusTool().handler(buildJsonObject { })

        assertTrue(result.text.contains("Battery Level:"))
        result.data["batteryPercent"]?.let { assertTrue(it.toString().toInt() in 0..100) }
        result.data["batterySource"]?.let {
            assertTrue(it.toString().trim('"') in setOf("battery_manager", "battery_changed"))
        }
    }

    @Test
    fun `open browser tool returns helpful error without url`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val factory = DeviceToolFactory(context)

        val result = factory.createOpenBrowserTool().handler(buildJsonObject { })

        assertTrue(result.isError)
        assertTrue(result.text.contains("No URL provided"))
    }

    @Test
    fun `open browser tool starts browsable intent`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val factory = DeviceToolFactory(context)

        val result =
            factory.createOpenBrowserTool().handler(
                buildJsonObject {
                    put("url", "https://developer.android.com")
                },
            )
        val startedIntent = shadowOf(context).nextStartedActivity

        assertTrue(!result.isError)
        assertEquals(Intent.ACTION_VIEW, startedIntent.action)
        assertEquals("https://developer.android.com", startedIntent.dataString)
        assertTrue(startedIntent.hasCategory(Intent.CATEGORY_BROWSABLE))
        assertTrue((startedIntent.flags and Intent.FLAG_ACTIVITY_NEW_TASK) != 0)
    }
}
