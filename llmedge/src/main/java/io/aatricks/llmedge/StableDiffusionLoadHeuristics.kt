package io.aatricks.llmedge

import android.app.ActivityManager
import android.content.Context
import io.aatricks.llmedge.model.ModelFileValidator

internal object StableDiffusionLoadHeuristics {
    fun computeEffectiveSequentialLoad(
        context: Context,
        resolvedModelPath: String,
        sequentialLoad: Boolean?,
        preferPerformanceMode: Boolean,
        activityManagerOverride: ActivityManager? = null,
    ): Pair<Boolean, Long> {
        val memoryInfo = ActivityManager.MemoryInfo()
        val activityManager =
            activityManagerOverride
                ?: (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager)
        activityManager.getMemoryInfo(memoryInfo)
        val totalRamGB = memoryInfo.totalMem / (1024L * 1024L * 1024L)

        if (sequentialLoad != null) {
            return sequentialLoad to 0L
        }

        val estimatedParamBytes =
            try {
                val devIdx = if (StableDiffusion.getVulkanDeviceCount() > 0) 0 else -1
                StableDiffusion.estimateModelParamsMemoryBytes(resolvedModelPath, devIdx)
            } catch (_: Throwable) {
                0L
            }

        val runtime = Runtime.getRuntime()
        val heapUsed = runtime.totalMemory() - runtime.freeMemory()
        val heapMax = runtime.maxMemory()
        val heapAvail = (heapMax - heapUsed).coerceAtLeast(0L)
        val sysAvail = memoryInfo.availMem

        val heapThresholdFactor = if (preferPerformanceMode) 0.9 else 0.75
        val sysThresholdFactor = if (preferPerformanceMode) 0.9 else 0.6

        val heapSeqNeeded =
            estimatedParamBytes > 0 &&
                estimatedParamBytes.toDouble() > heapAvail.toDouble() * heapThresholdFactor
        val sysSeqNeeded =
            estimatedParamBytes > 0 &&
                estimatedParamBytes.toDouble() > sysAvail.toDouble() * sysThresholdFactor
        val lowRamHint = totalRamGB < 8

        return (lowRamHint || heapSeqNeeded || sysSeqNeeded) to estimatedParamBytes
    }

    fun validateResolvedAssets(
        modelPath: String,
        vaePath: String?,
        t5xxlPath: String?,
        taesdPath: String?,
        loraModelDir: String?,
    ) {
        ModelFileValidator.requireReadableFile(modelPath, "Stable Diffusion model")
        vaePath?.let { ModelFileValidator.requireReadableFile(it, "Stable Diffusion VAE") }
        t5xxlPath?.let { ModelFileValidator.requireReadableFile(it, "Stable Diffusion text encoder") }
        taesdPath?.let { ModelFileValidator.requireReadableFile(it, "Stable Diffusion TAE") }
        loraModelDir?.let { ModelFileValidator.requireReadableDirectory(it, "LoRA model directory") }
    }
}