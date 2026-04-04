package io.aatricks.llmedge.image.diffusion

import android.content.Context
import io.aatricks.llmedge.image.diffusion.internal.StableDiffusionLoader
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.CpuTopology
import kotlinx.coroutines.CoroutineDispatcher

internal object StableDiffusionStaticApiSupport {
    val diffusionDispatcher: CoroutineDispatcher
        get() = StableDiffusionCompanionSupport.diffusionDispatcher

    fun computeEffectiveSequentialLoad(
        context: Context,
        resolvedModelPath: String,
        sequentialLoad: Boolean?,
        preferPerformanceMode: Boolean,
        activityManagerOverride: android.app.ActivityManager?,
    ): Pair<Boolean, Long> =
        StableDiffusionCompanionSupport.computeEffectiveSequentialLoad(
            context = context,
            resolvedModelPath = resolvedModelPath,
            sequentialLoad = sequentialLoad,
            preferPerformanceMode = preferPerformanceMode,
            activityManagerOverride = activityManagerOverride,
        )

    fun isNativeLibraryLoaded(nativeCheckBindings: () -> Boolean): Boolean =
        StableDiffusionCompanionSupport.isNativeLibraryLoaded(nativeCheckBindings)

    fun enableNativeBridgeForTests() {
        StableDiffusionCompanionSupport.enableNativeBridgeForTests()
    }

    fun overrideNativeBridgeForTests(provider: (StableDiffusion) -> StableDiffusion.NativeBridge) {
        StableDiffusionCompanionSupport.overrideNativeBridgeForTests(provider)
    }

    fun resetNativeBridgeForTests() {
        StableDiffusionCompanionSupport.resetNativeBridgeForTests()
    }

    fun getVulkanDeviceCount(nativeCall: () -> Int): Int =
        StableDiffusionCompanionSupport.getVulkanDeviceCount(nativeCall)

    fun getVulkanDeviceMemory(nativeCall: () -> LongArray?): LongArray? =
        StableDiffusionCompanionSupport.getVulkanDeviceMemory(nativeCall)

    fun getVulkanDeviceDescription(nativeCall: () -> String?): String? =
        StableDiffusionCompanionSupport.getVulkanDeviceDescription(nativeCall)

    fun estimateModelParamsMemoryBytes(nativeCall: () -> Long): Long =
        StableDiffusionCompanionSupport.estimateModelParamsMemoryBytes(nativeCall)

    fun checkBindings(nativeCall: () -> Boolean): Boolean =
        StableDiffusionCompanionSupport.checkBindings(nativeCall)

    fun isOpenClAvailable(nativeCall: () -> Boolean): Boolean =
        StableDiffusionCompanionSupport.isOpenClAvailable(nativeCall)

    fun supportNativeCreate(
        modelPath: String,
        vaePath: String?,
        t5xxlPath: String?,
        taesdPath: String?,
        nThreads: Int,
        enableOpenCl: Boolean,
        useVulkan: Boolean,
        offloadToCpu: Boolean,
        keepClipOnCpu: Boolean,
        keepVaeOnCpu: Boolean,
        flashAttn: Boolean,
        vaeDecodeOnly: Boolean,
        flowShift: Float,
        loraModelDir: String?,
        loraApplyMode: Int,
        nativeCreate: (
            String,
            String?,
            String?,
            String?,
            Int,
            Boolean,
            Boolean,
            Boolean,
            Boolean,
            Boolean,
            Boolean,
            Boolean,
            Float,
            String?,
            Int,
        ) -> Long,
    ): Long =
        nativeCreate(
            modelPath,
            vaePath,
            t5xxlPath,
            taesdPath,
            nThreads,
            enableOpenCl,
            useVulkan,
            offloadToCpu,
            keepClipOnCpu,
            keepVaeOnCpu,
            flashAttn,
            vaeDecodeOnly,
            flowShift,
            loraModelDir,
            loraApplyMode,
        )

    suspend fun load(
        context: Context,
        modelId: String? = null,
        filename: String? = null,
        modelPath: String? = null,
        vaePath: String? = null,
        t5xxlPath: String? = null,
        taesdPath: String? = null,
        nThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
        offloadToCpu: Boolean = false,
        keepClipOnCpu: Boolean = false,
        keepVaeOnCpu: Boolean = false,
        flashAttn: Boolean = true,
        vaeDecodeOnly: Boolean = true,
        sequentialLoad: Boolean? = null,
        allowVulkan: Boolean = true,
        forceVulkan: Boolean = false,
        preferPerformanceMode: Boolean = false,
        token: String? = null,
        forceDownload: Boolean = false,
        flowShift: Float = Float.POSITIVE_INFINITY,
        loraModelDir: String? = null,
        loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
    ): StableDiffusion = StableDiffusionLoader.load(
        context = context,
        modelId = modelId,
        filename = filename,
        modelPath = modelPath,
        vaePath = vaePath,
        t5xxlPath = t5xxlPath,
        taesdPath = taesdPath,
        nThreads = nThreads,
        offloadToCpu = offloadToCpu,
        keepClipOnCpu = keepClipOnCpu,
        keepVaeOnCpu = keepVaeOnCpu,
        flashAttn = flashAttn,
        vaeDecodeOnly = vaeDecodeOnly,
        sequentialLoad = sequentialLoad,
        allowVulkan = allowVulkan,
        forceVulkan = forceVulkan,
        preferPerformanceMode = preferPerformanceMode,
        token = token,
        forceDownload = forceDownload,
        flowShift = flowShift,
        loraModelDir = loraModelDir,
        loraApplyMode = loraApplyMode,
    )

    suspend fun loadWithRuntimeBackend(
        context: Context,
        modelId: String? = null,
        filename: String? = null,
        modelPath: String? = null,
        vaePath: String? = null,
        t5xxlPath: String? = null,
        taesdPath: String? = null,
        nThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
        offloadToCpu: Boolean = false,
        keepClipOnCpu: Boolean = false,
        keepVaeOnCpu: Boolean = false,
        flashAttn: Boolean = true,
        vaeDecodeOnly: Boolean = true,
        sequentialLoad: Boolean? = null,
        preferPerformanceMode: Boolean = false,
        token: String? = null,
        forceDownload: Boolean = false,
        flowShift: Float = Float.POSITIVE_INFINITY,
        loraModelDir: String? = null,
        loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
        preferredBackend: ComputeBackend,
    ): StableDiffusion = StableDiffusionLoader.loadWithRuntimeBackend(
        context = context,
        modelId = modelId,
        filename = filename,
        modelPath = modelPath,
        vaePath = vaePath,
        t5xxlPath = t5xxlPath,
        taesdPath = taesdPath,
        nThreads = nThreads,
        offloadToCpu = offloadToCpu,
        keepClipOnCpu = keepClipOnCpu,
        keepVaeOnCpu = keepVaeOnCpu,
        flashAttn = flashAttn,
        vaeDecodeOnly = vaeDecodeOnly,
        sequentialLoad = sequentialLoad,
        preferPerformanceMode = preferPerformanceMode,
        token = token,
        forceDownload = forceDownload,
        flowShift = flowShift,
        loraModelDir = loraModelDir,
        loraApplyMode = loraApplyMode,
        preferredBackend = preferredBackend,
    )

    suspend fun loadFromHuggingFace(
        context: Context,
        modelId: String,
        filename: String? = null,
        taesdPath: String? = null,
        nThreads: Int = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
        offloadToCpu: Boolean = false,
        keepClipOnCpu: Boolean = false,
        keepVaeOnCpu: Boolean = false,
        flashAttn: Boolean = true,
        vaeDecodeOnly: Boolean = true,
        sequentialLoad: Boolean? = null,
        allowVulkan: Boolean = true,
        forceVulkan: Boolean = false,
        preferPerformanceMode: Boolean = false,
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = true,
        flowShift: Float = Float.POSITIVE_INFINITY,
        loraModelDir: String? = null,
        loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
        onProgress: ((name: String, downloaded: Long, total: Long?) -> Unit)? = null,
    ): StableDiffusion = StableDiffusionLoader.loadFromHuggingFace(
        context = context,
        modelId = modelId,
        filename = filename,
        taesdPath = taesdPath,
        nThreads = nThreads,
        offloadToCpu = offloadToCpu,
        keepClipOnCpu = keepClipOnCpu,
        keepVaeOnCpu = keepVaeOnCpu,
        flashAttn = flashAttn,
        vaeDecodeOnly = vaeDecodeOnly,
        sequentialLoad = sequentialLoad,
        allowVulkan = allowVulkan,
        forceVulkan = forceVulkan,
        preferPerformanceMode = preferPerformanceMode,
        token = token,
        forceDownload = forceDownload,
        preferSystemDownloader = preferSystemDownloader,
        flowShift = flowShift,
        loraModelDir = loraModelDir,
        loraApplyMode = loraApplyMode,
        onProgress = onProgress,
    )
}
