package io.aatricks.llmedge.tools

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Factory for creating real-world Android tools for the LLM.
 */
class DeviceToolFactory(private val context: Context) {
    fun createDefaultTools(includeActions: Boolean = true): List<Tool> =
        buildList {
            add(createGetTimeTool())
            add(createGetBatteryStatusTool())
            add(createGetDeviceInfoTool())
            if (includeActions) {
                add(createOpenBrowserTool())
            }
        }

    fun createGetTimeTool(): Tool =
        Tool(
            name = "get_current_time",
            description = "Returns the current date and time in a human-readable format.",
            handler = {
                val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                val timestamp = sdf.format(Date())
                ToolResult.success(
                    text = "Current Date and Time: $timestamp",
                    data = jsonObject("timestamp" to timestamp),
                )
            },
        )

    fun createGetDeviceInfoTool(): Tool =
        Tool(
            name = "get_device_info",
            description = "Returns basic information about the device hardware.",
            handler = {
                val text =
                    "Manufacturer: ${Build.MANUFACTURER}, Model: ${Build.MODEL}, Android Version: ${Build.VERSION.RELEASE}"
                ToolResult.success(
                    text = text,
                    data =
                        buildJsonObject {
                            put("manufacturer", Build.MANUFACTURER)
                            put("model", Build.MODEL)
                            put("androidVersion", Build.VERSION.RELEASE)
                        },
                )
            },
        )

    fun createGetBatteryStatusTool(): Tool =
        Tool(
            name = "get_battery_status",
            description = "Returns the current battery level and whether the device is charging.",
            handler = {
                val intentFilter = android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED)
                val batteryStatus: Intent? = context.registerReceiver(null, intentFilter)

                val status: Int = batteryStatus?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
                val isCharging =
                    status == BatteryManager.BATTERY_STATUS_CHARGING ||
                        status == BatteryManager.BATTERY_STATUS_FULL

                val level: Int = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
                val scale: Int = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
                val batteryPct = if (level != -1 && scale != -1) (level * 100 / scale.toFloat()) else -1.0f

                ToolResult.success(
                    text = "Battery Level: ${batteryPct}%, Charging: $isCharging",
                    data =
                        buildJsonObject {
                            put("batteryPercent", batteryPct)
                            put("isCharging", isCharging)
                        },
                )
            },
        )

    fun createOpenBrowserTool(): Tool =
        Tool(
            name = "open_browser",
            description = "Opens the provided URL in the system's default web browser.",
            kind = ToolKind.ACTION,
            schema =
                ToolSchema(
                    parameters =
                        mapOf(
                            "url" to
                                ToolParameter(
                                    type = ToolParameterType.STRING,
                                    description = "The full URL to open, e.g., 'https://www.google.com'",
                                ),
                        ),
                ),
            handler = { args ->
                openBrowser(args)
            },
        )

    private fun openBrowser(args: JsonObject): ToolResult {
        val url = args["url"]?.jsonPrimitive?.contentOrNull
            ?: return ToolResult.error("No URL provided.", jsonObject("code" to "missing_url"))

        return try {
            val intent =
                Intent(Intent.ACTION_VIEW, Uri.parse(url)).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
            context.startActivity(intent)
            ToolResult.success(
                text = "Successfully opened $url in the browser.",
                data = jsonObject("url" to url),
            )
        } catch (e: Exception) {
            ToolResult.error(
                text = "Error: Could not open URL - ${e.message}",
                data =
                    buildJsonObject {
                        put("url", url)
                        put("code", "open_failed")
                        put("message", e.message ?: "Unknown error.")
                    },
            )
        }
    }
}

private fun jsonObject(vararg fields: Pair<String, Any?>): JsonObject =
    buildJsonObject {
        fields.forEach { (key, value) ->
            when (value) {
                null -> Unit
                is String -> put(key, value)
                is Boolean -> put(key, value)
                is Int -> put(key, value)
                is Long -> put(key, value)
                is Float -> put(key, value)
                is Double -> put(key, value)
                is Short -> put(key, value.toInt())
                is Byte -> put(key, value.toInt())
                else -> put(key, value.toString())
            }
        }
    }
