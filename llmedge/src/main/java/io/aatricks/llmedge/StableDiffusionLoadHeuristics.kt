package io.aatricks.llmedge

import android.app.ActivityManager
import android.content.Context
import io.aatricks.llmedge.model.ModelFileValidator

internal object StableDiffusionLoadHeuristics {
    internal data class MemorySnapshot(
        val totalRamGb: Long,
        val availableSystemBytes: Long,
        val availableHeapBytes: Long,
    )

    internal data class LoadPlan(
        val effectiveSequentialLoad: Boolean,
        val estimatedParamBytes: Long,
        val effectiveOffloadToCpu: Boolean,
        val effectiveKeepClipOnCpu: Boolean,
        val effectiveKeepVaeOnCpu: Boolean,
        val chosenDevice: Int,
        val estimatedDeviceParamsBytes: Long,
        val freeVulkanBytes: Long,
        val memorySnapshot: MemorySnapshot,
    )

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

    fun captureMemorySnapshot(
        context: Context,
        activityManagerOverride: ActivityManager? = null,
    ): MemorySnapshot {
        val memoryInfo = ActivityManager.MemoryInfo()
        val activityManager =
            activityManagerOverride
                ?: (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager)
        activityManager.getMemoryInfo(memoryInfo)

        val runtime = Runtime.getRuntime()
        val heapUsed = runtime.totalMemory() - runtime.freeMemory()
        val heapAvail = (runtime.maxMemory() - heapUsed).coerceAtLeast(0L)

        return MemorySnapshot(
            totalRamGb = memoryInfo.totalMem / (1024L * 1024L * 1024L),
            availableSystemBytes = memoryInfo.availMem,
            availableHeapBytes = heapAvail,
        )
    }

    fun warnIfLargeModelOnLowRam(
        metadata: VideoModelMetadata,
        memorySnapshot: MemorySnapshot,
        warn: (String) -> Unit,
    ) {
        if (metadata.parameterCount == "5B" && memorySnapshot.totalRamGb < 8) {
            warn(
                "Loading 5B model on device with ${memorySnapshot.totalRamGb}GB RAM. " +
                    "Consider using 1.3B variant for better performance. Generation may be slow or fail with OOM.",
            )
        }
    }

    fun planLoad(
        context: Context,
        resolvedModelPath: String,
        sequentialLoad: Boolean?,
        preferPerformanceMode: Boolean,
        offloadToCpu: Boolean,
        keepClipOnCpu: Boolean,
        keepVaeOnCpu: Boolean,
        forceVulkan: Boolean,
        activityManagerOverride: ActivityManager? = null,
        getVulkanDeviceCount: () -> Int = { StableDiffusion.getVulkanDeviceCount() },
        getVulkanDeviceMemory: (Int) -> LongArray? = { StableDiffusion.getVulkanDeviceMemory(it) },
        estimateModelParamsMemoryBytes: (String, Int) -> Long = { modelPath, deviceIndex ->
            StableDiffusion.estimateModelParamsMemoryBytes(modelPath, deviceIndex)
        },
    ): LoadPlan {
        val memorySnapshot = captureMemorySnapshot(context, activityManagerOverride)
        val (effectiveSequentialLoad, estimatedParamBytes) =
            computeEffectiveSequentialLoad(
                context = context,
                resolvedModelPath = resolvedModelPath,
                sequentialLoad = sequentialLoad,
                preferPerformanceMode = preferPerformanceMode,
                activityManagerOverride = activityManagerOverride,
            )

        var effectiveOffloadToCpu = offloadToCpu
        var effectiveKeepClipOnCpu = keepClipOnCpu
        var effectiveKeepVaeOnCpu = keepVaeOnCpu

        if (effectiveSequentialLoad && sequentialLoad == null) {
            effectiveOffloadToCpu = true
            effectiveKeepClipOnCpu = true
            effectiveKeepVaeOnCpu = true
        }

        var chosenDevice = -1
        var estimatedDeviceParamsBytes = 0L
        var freeVulkanBytes = 0L

        if (!effectiveOffloadToCpu) {
            val vulkanDevices = getVulkanDeviceCount()
            if (vulkanDevices > 0) {
                var maxTotal = 0L
                for (device in 0 until vulkanDevices) {
                    val memory = getVulkanDeviceMemory(device)
                    if (memory != null && memory.size >= 2 && memory[1] > maxTotal) {
                        maxTotal = memory[1]
                        chosenDevice = device
                    }
                }

                if (chosenDevice >= 0) {
                    estimatedDeviceParamsBytes = estimateModelParamsMemoryBytes(resolvedModelPath, chosenDevice)
                    if (estimatedDeviceParamsBytes > 0) {
                        val memory = getVulkanDeviceMemory(chosenDevice)
                        if (memory != null && memory.size >= 2) {
                            freeVulkanBytes = memory[0]
                            val threshold = 0.9
                            if (!forceVulkan && estimatedDeviceParamsBytes.toDouble() > freeVulkanBytes.toDouble() * threshold) {
                                effectiveOffloadToCpu = true
                            }
                        }
                    }
                }
            }
        }

        return LoadPlan(
            effectiveSequentialLoad = effectiveSequentialLoad,
            estimatedParamBytes = estimatedParamBytes,
            effectiveOffloadToCpu = effectiveOffloadToCpu,
            effectiveKeepClipOnCpu = effectiveKeepClipOnCpu,
            effectiveKeepVaeOnCpu = effectiveKeepVaeOnCpu,
            chosenDevice = chosenDevice,
            estimatedDeviceParamsBytes = estimatedDeviceParamsBytes,
            freeVulkanBytes = freeVulkanBytes,
            memorySnapshot = memorySnapshot,
        )
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