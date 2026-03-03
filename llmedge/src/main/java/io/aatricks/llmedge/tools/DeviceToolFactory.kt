package io.aatricks.llmedge.tools

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Factory for creating real-world Android tools for the LLM.
 */
class DeviceToolFactory(private val context: Context) {

    /**
     * Tool to get the current date and time.
     */
    fun createGetTimeTool() = Tool(
        name = "get_current_time",
        description = "Returns the current date and time in a human-readable format.",
        parameters = emptyMap(),
        execute = {
            val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
            "Current Date and Time: ${sdf.format(Date())}"
        }
    )

    /**
     * Tool to get device information like model and manufacturer.
     */
    fun createGetDeviceInfoTool() = Tool(
        name = "get_device_info",
        description = "Returns basic information about the device hardware.",
        parameters = emptyMap(),
        execute = {
            "Manufacturer: ${Build.MANUFACTURER}, Model: ${Build.MODEL}, Android Version: ${Build.VERSION.RELEASE}"
        }
    )

    /**
     * Tool to check the current battery level and charging status.
     */
    fun createGetBatteryStatusTool() = Tool(
        name = "get_battery_status",
        description = "Returns the current battery level and whether the device is charging.",
        parameters = emptyMap(),
        execute = {
            val intentFilter = android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            val batteryStatus: Intent? = context.registerReceiver(null, intentFilter)
            
            val status: Int = batteryStatus?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            val isCharging: Boolean = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                                     status == BatteryManager.BATTERY_STATUS_FULL
            
            val level: Int = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale: Int = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
            val batteryPct = if (level != -1 && scale != -1) (level * 100 / scale.toFloat()) else -1.0f
            
            "Battery Level: ${batteryPct}%, Charging: $isCharging"
        }
    )

    /**
     * Tool to open a specified URL in the device's default web browser.
     */
    fun createOpenBrowserTool() = Tool(
        name = "open_browser",
        description = "Opens the provided URL in the system's default web browser.",
        parameters = mapOf(
            "url" to ParameterDescription("string", "The full URL to open, e.g., 'https://www.google.com'")
        ),
        execute = { args ->
            val url = args["url"] as? String
            if (url != null) {
                try {
                    val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url)).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    context.startActivity(intent)
                    "Successfully opened $url in the browser."
                } catch (e: Exception) {
                    "Error: Could not open URL - ${e.message}"
                }
            } else {
                "Error: No URL provided."
            }
        }
    )
}
