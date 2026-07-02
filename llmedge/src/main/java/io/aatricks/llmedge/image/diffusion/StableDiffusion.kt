/*
 * Copyright (C) 2024 LLMEdge Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.aatricks.llmedge.image.diffusion

import io.aatricks.llmedge.runtime.CpuTopology
import io.aatricks.llmedge.runtime.ComputeBackend

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.UnsupportedModelException
import io.aatricks.llmedge.image.diffusion.internal.StableDiffusionExecutor
import io.aatricks.llmedge.image.diffusion.internal.StableDiffusionLoader
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.withLock
import kotlin.jvm.JvmName

/**
 * Advanced wrapper around the Stable Diffusion / Wan native runtime.
 *
 * Most application code should prefer `LLMEdge.create(...).image` so model resolution, cache
 * ownership, backend fallback, and cancellation all stay on the standardized facade path.
 */
class StableDiffusion internal constructor(
    private val handle: Long,
    private val nativeLibrarySupport: StableDiffusionNativeLibrarySupport = StableDiffusionCompanionSupport.currentNativeLibrarySupport(),
) : AutoCloseable {
    // Preserve the legacy reflective test seam while routing through explicit native loading.
    private constructor(handle: Long) : this(handle, StableDiffusionCompanionSupport.currentNativeLibrarySupport())

    init {
        nativeLibrarySupport.ensureLoaded()
    }

    private val runtimeState = StableDiffusionState(handle)
    private val nativeBridge: NativeBridge = StableDiffusionCompanionSupport.createNativeBridge(this)
    // Legacy reflective tests still reach into these fields directly, so keep thin mirrors.
    private val cancellationRequested = runtimeState.cancellationRequested
    private var cachedProgressCallback: VideoProgressCallback? = null

    internal interface NativeBridge : StableDiffusionNativeBridgeContract

    /** Generates an image from text. */
    fun txt2img(
            prompt: String,
            negative: String = "",
            width: Int = 512,
            height: Int = 512,
            steps: Int = 20,
            cfg: Float = 7.0f,
            seed: Long = 42L,
            vaeTiling: Boolean = true,
            easyCacheEnabled: Boolean = false,
            easyCacheReuseThreshold: Float = 0.2f,
            easyCacheStartPercent: Float = 0.15f,
            easyCacheEndPercent: Float = 0.95f
    ): ByteArray? {
        return nativeBridge.txt2img(
                handle,
                prompt,
                negative,
                width,
                height,
                steps,
                cfg,
                seed,
                vaeTiling,
                easyCacheEnabled,
                easyCacheReuseThreshold,
                easyCacheStartPercent,
                easyCacheEndPercent
        )
    }

    fun txt2vid(
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            videoFrames: Int,
            steps: Int,
            cfg: Float,
            seed: Long,
            sampleMethod: SampleMethod,
            scheduler: Scheduler,
            strength: Float,
            initImage: ByteArray?,
            initWidth: Int,
            initHeight: Int,
            vaceStrength: Float = 1.0f,
            easyCacheEnabled: Boolean = false,
            easyCacheReuseThreshold: Float = 0.2f,
            easyCacheStartPercent: Float = 0.15f,
            easyCacheEndPercent: Float = 0.95f
    ): Array<ByteArray>? {
        return nativeBridge.txt2vid(
                handle,
                prompt,
                negative,
                width,
                height,
                videoFrames,
                steps,
                cfg,
                seed,
                sampleMethod,
                scheduler,
                strength,
                initImage,
                initWidth,
                initHeight,
                vaceStrength,
                easyCacheEnabled,
                easyCacheReuseThreshold,
                easyCacheStartPercent,
                easyCacheEndPercent
        )
    }

    fun txt2ImgWithPrecomputedCondition(
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            steps: Int,
            cfg: Float,
            seed: Long,
            cond: PrecomputedCondition?,
            uncond: PrecomputedCondition?,
            easyCacheEnabled: Boolean = false,
            easyCacheReuseThreshold: Float = 0.2f,
            easyCacheStartPercent: Float = 0.15f,
            easyCacheEndPercent: Float = 0.95f,
    ): ByteArray? {
        return nativeBridge.txt2ImgWithPrecomputedCondition(
                handle,
                prompt,
                negative,
                width,
                height,
                steps,
                cfg,
                seed,
                cond,
                uncond,
                easyCacheEnabled,
                easyCacheReuseThreshold,
                easyCacheStartPercent,
                easyCacheEndPercent
        )
    }

    companion object {
        private const val BYTES_IN_MB = 1024L * 1024L
        private const val MEMORY_PRESSURE_THRESHOLD = 0.85f
        val diffusionDispatcher: CoroutineDispatcher = StableDiffusionCompanionSupport.diffusionDispatcher

        // Kept as a reflective seam for heuristic tests that validate the combined load policy.
        @JvmStatic
        private fun computeEffectiveSequentialLoad(
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

        /**
         * Helper for tests and runtime checks to verify whether the native sdcpp library is
         * implemented and correctly linked. This attempts to call the JNI `nativeCheckBindings`
         * method and returns false if the library is not present.
         */
        @JvmStatic
        fun isNativeLibraryLoaded(): Boolean {
            return StableDiffusionCompanionSupport.isNativeLibraryLoaded(::nativeCheckBindings)
        }

        internal fun enableNativeBridgeForTests() {
            StableDiffusionCompanionSupport.enableNativeBridgeForTests()
        }

        internal fun overrideNativeBridgeForTests(provider: (StableDiffusion) -> NativeBridge) {
            StableDiffusionCompanionSupport.overrideNativeBridgeForTests(provider)
        }

        internal fun resetNativeBridgeForTests() {
            StableDiffusionCompanionSupport.resetNativeBridgeForTests()
        }

        /**
         * Get the number of Vulkan devices available on this system
         * @return Number of Vulkan-capable devices, or 0 if Vulkan is not available
         */
        @JvmStatic
        fun getVulkanDeviceCount(): Int {
            return StableDiffusionCompanionSupport.getVulkanDeviceCount(::nativeGetVulkanDeviceCount)
        }

        /**
         * Get Vulkan device memory information
         * @param deviceIndex Index of the device to query (default 0)
         * @return LongArray with [freeMemory, totalMemory] in bytes, or null if unavailable
         */
        @JvmStatic
        fun getVulkanDeviceMemory(deviceIndex: Int = 0): LongArray? {
            return StableDiffusionCompanionSupport.getVulkanDeviceMemory {
                nativeGetVulkanDeviceMemory(deviceIndex)
            }
        }

        /**
         * Get a human-readable Vulkan device description.
         */
        @JvmStatic
        fun getVulkanDeviceDescription(deviceIndex: Int = 0): String? {
            return StableDiffusionCompanionSupport.getVulkanDeviceDescription {
                nativeGetVulkanDeviceDescription(deviceIndex)
            }
        }

        /**
         * Public wrapper that attempts to estimate the model parameter memory (in bytes) for a
         * model path on a given device. Returns 0 on failure or if the native estimation is not
         * available. This is a convenience helper used by higher-level managers to compute cache
         * sizes and decide on offload heuristics.
         */
        @JvmStatic
        fun estimateModelParamsMemoryBytes(modelPath: String, deviceIndex: Int = 0): Long {
            return StableDiffusionCompanionSupport.estimateModelParamsMemoryBytes {
                nativeEstimateModelParamsMemory(modelPath, deviceIndex)
            }
        }

        @JvmStatic
        fun checkBindings(): Boolean {
            return StableDiffusionCompanionSupport.checkBindings(::nativeCheckBindings)
        }

        @JvmStatic
        fun isOpenClAvailable(): Boolean {
            return StableDiffusionCompanionSupport.isOpenClAvailable(::nativeIsOpenClAvailable)
        }

        internal fun supportNativeCreate(request: StableDiffusionNativeLoadRequest): Long =
            nativeCreate(
                request.modelPath,
                request.vaePath,
                request.t5xxlPath,
                request.taesdPath,
                request.diffusionModelPath,
                request.llmPath,
                request.nThreads,
                request.enableOpenCl,
                request.useVulkan,
                request.offloadToCpu,
                request.keepClipOnCpu,
                request.keepVaeOnCpu,
                request.flashAttn,
                request.vaeDecodeOnly,
                request.flowShift,
                request.loraModelDir,
                request.loraApplyMode.id,
            )

        @JvmStatic
        private external fun nativeCreate(
                modelPath: String,
                vaePath: String?,
                t5xxlPath: String?,
                taesdPath: String?,
                diffusionModelPath: String?,
                llmPath: String?,
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
                loraApplyMode: Int
        ): Long

        @JvmStatic
        private external fun nativeGetVulkanDeviceCount(): Int

        @JvmStatic
        private external fun nativeGetVulkanDeviceMemory(deviceIndex: Int): LongArray?

        @JvmStatic
        private external fun nativeGetVulkanDeviceDescription(deviceIndex: Int): String?

        @JvmStatic
        private external fun nativeEstimateModelParamsMemory(
                modelPath: String,
                deviceIndex: Int
        ): Long

        @JvmStatic
        private external fun nativeEstimateModelParamsMemoryDetailed(
                modelPath: String,
                deviceIndex: Int
        ): LongArray?

        @JvmStatic
        private external fun nativeCheckBindings(): Boolean

        @JvmStatic
        private external fun nativeIsOpenClAvailable(): Boolean

        suspend fun load(
                context: Context,
                modelId: String? = null,
                filename: String? = null,
                modelPath: String? = null,
                vaePath: String? = null,
                t5xxlPath: String? = null,
                taesdPath: String? = null,
                diffusionModelPath: String? = null,
                llmPath: String? = null,
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
        ): StableDiffusion =
            StableDiffusionLoader.load(
                context = context,
                request =
                    createLoadRequest(
                        modelId = modelId,
                        filename = filename,
                        modelPath = modelPath,
                        vaePath = vaePath,
                        t5xxlPath = t5xxlPath,
                        taesdPath = taesdPath,
                        diffusionModelPath = diffusionModelPath,
                        llmPath = llmPath,
                        nThreads = nThreads,
                        offloadToCpu = offloadToCpu,
                        keepClipOnCpu = keepClipOnCpu,
                        keepVaeOnCpu = keepVaeOnCpu,
                        flashAttn = flashAttn,
                        vaeDecodeOnly = vaeDecodeOnly,
                        sequentialLoad = sequentialLoad,
                        allowOpenCl = true,
                        allowVulkan = allowVulkan,
                        forceVulkan = forceVulkan,
                        preferPerformanceMode = preferPerformanceMode,
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = true,
                        flowShift = flowShift,
                        loraModelDir = loraModelDir,
                        loraApplyMode = loraApplyMode,
                        preferredBackend = null,
                        allowBackendFallbackToCpu = true,
                    ),
            )

        internal suspend fun loadWithRuntimeBackend(
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
                // Appended after preferredBackend so the existing positional argument order (relied
                // on by mocks/tests) is preserved.
                diffusionModelPath: String? = null,
                llmPath: String? = null,
        ): StableDiffusion =
            StableDiffusionLoader.load(
                context = context,
                request =
                    createLoadRequest(
                        modelId = modelId,
                        filename = filename,
                        modelPath = modelPath,
                        vaePath = vaePath,
                        t5xxlPath = t5xxlPath,
                        taesdPath = taesdPath,
                        diffusionModelPath = diffusionModelPath,
                        llmPath = llmPath,
                        nThreads = nThreads,
                        offloadToCpu = offloadToCpu,
                        keepClipOnCpu = keepClipOnCpu,
                        keepVaeOnCpu = keepVaeOnCpu,
                        flashAttn = flashAttn,
                        vaeDecodeOnly = vaeDecodeOnly,
                        sequentialLoad = sequentialLoad,
                        allowOpenCl = preferredBackend == ComputeBackend.OPENCL,
                        allowVulkan = preferredBackend == ComputeBackend.VULKAN,
                        forceVulkan = preferredBackend == ComputeBackend.VULKAN,
                        preferPerformanceMode = preferPerformanceMode,
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = true,
                        flowShift = flowShift,
                        loraModelDir = loraModelDir,
                        loraApplyMode = loraApplyMode,
                        preferredBackend = preferredBackend,
                        allowBackendFallbackToCpu = false,
                    ),
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
        ): StableDiffusion =
            StableDiffusionLoader.loadFromHuggingFace(
                context = context,
                request =
                    createLoadRequest(
                        modelId = modelId,
                        filename = filename,
                        modelPath = null,
                        vaePath = null,
                        t5xxlPath = null,
                        taesdPath = taesdPath,
                        nThreads = nThreads,
                        offloadToCpu = offloadToCpu,
                        keepClipOnCpu = keepClipOnCpu,
                        keepVaeOnCpu = keepVaeOnCpu,
                        flashAttn = flashAttn,
                        vaeDecodeOnly = vaeDecodeOnly,
                        sequentialLoad = sequentialLoad,
                        allowOpenCl = true,
                        allowVulkan = allowVulkan,
                        forceVulkan = forceVulkan,
                        preferPerformanceMode = preferPerformanceMode,
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = preferSystemDownloader,
                        flowShift = flowShift,
                        loraModelDir = loraModelDir,
                        loraApplyMode = loraApplyMode,
                        preferredBackend = null,
                        allowBackendFallbackToCpu = true,
                    ),
                onProgress = onProgress,
            )

        private fun createLoadRequest(
            modelId: String?,
            filename: String?,
            modelPath: String?,
            vaePath: String?,
            t5xxlPath: String?,
            taesdPath: String?,
            nThreads: Int,
            diffusionModelPath: String? = null,
            llmPath: String? = null,
            offloadToCpu: Boolean,
            keepClipOnCpu: Boolean,
            keepVaeOnCpu: Boolean,
            flashAttn: Boolean,
            vaeDecodeOnly: Boolean,
            sequentialLoad: Boolean?,
            allowOpenCl: Boolean,
            allowVulkan: Boolean,
            forceVulkan: Boolean,
            preferPerformanceMode: Boolean,
            token: String?,
            forceDownload: Boolean,
            preferSystemDownloader: Boolean,
            flowShift: Float,
            loraModelDir: String?,
            loraApplyMode: LoraApplyMode,
            preferredBackend: ComputeBackend?,
            allowBackendFallbackToCpu: Boolean,
        ): StableDiffusionLoadRequest =
            StableDiffusionLoadRequest(
                assets =
                    StableDiffusionAssetRequest(
                        modelId = modelId,
                        filename = filename,
                        modelPath = modelPath,
                        vaePath = vaePath,
                        t5xxlPath = t5xxlPath,
                        taesdPath = taesdPath,
                        diffusionModelPath = diffusionModelPath,
                        llmPath = llmPath,
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = preferSystemDownloader,
                        loraModelDir = loraModelDir,
                    ),
                runtime =
                    StableDiffusionRuntimeRequest(
                        nThreads = nThreads,
                        offloadToCpu = offloadToCpu,
                        keepClipOnCpu = keepClipOnCpu,
                        keepVaeOnCpu = keepVaeOnCpu,
                        flashAttn = flashAttn,
                        vaeDecodeOnly = vaeDecodeOnly,
                        sequentialLoad = sequentialLoad,
                        preferPerformanceMode = preferPerformanceMode,
                        flowShift = flowShift,
                        loraApplyMode = loraApplyMode,
                    ),
                backend =
                    StableDiffusionBackendRequest(
                        allowOpenCl = allowOpenCl,
                        allowVulkan = allowVulkan,
                        forceVulkan = forceVulkan,
                        preferredBackend = preferredBackend,
                        allowBackendFallbackToCpu = allowBackendFallbackToCpu,
                    ),
            )

        internal fun supportLogWarning(message: String) = AndroidLogAdapter.w("StableDiffusion", message)

        internal fun supportIsNativeLibraryAvailable(): Boolean =
            StableDiffusionCompanionSupport.supportIsNativeLibraryAvailable()

        internal fun supportNativeBridgeOverriddenForTests(): Boolean =
            StableDiffusionCompanionSupport.supportNativeBridgeOverriddenForTests()

    }

    internal fun updateModelMetadata(metadata: VideoModelMetadata?) {
        runtimeState.modelMetadata = metadata
        runtimeState.easyCacheSupported =
            if (!supportIsNativeLibraryAvailable() || supportNativeBridgeOverriddenForTests()) {
                metadata?.let(StableDiffusionMetadataSupport::supportsEasyCache)
            } else {
                null
            }
    }

    fun isVideoModel(): Boolean = StableDiffusionFacadeOperations.isVideoModel(this)

    suspend fun txt2vid(
            params: VideoGenerateParams,
            onProgress: VideoProgressCallback? = null,
        ): List<Bitmap> = StableDiffusionFacadeOperations.txt2vid(this, params, onProgress)

    fun setProgressCallback(callback: VideoProgressCallback?) =
        StableDiffusionFacadeOperations.setProgressCallback(this, callback)

    fun cancelGeneration() = StableDiffusionFacadeOperations.cancelGeneration(this)

    fun getLastGenerationMetrics(): GenerationMetrics? = runtimeState.lastGenerationMetrics

    internal val bridge: NativeBridge
        get() = nativeBridge

    internal val state: StableDiffusionState
        get() = runtimeState

    internal fun beginImageRequestTrace(requestId: Long) =
        StableDiffusionFacadeOperations.beginImageRequestTrace(this, requestId)

    internal fun traceImagePhase(
        phase: ImageGenerationPhase,
        detail: String? = null,
        throwable: Throwable? = null,
    ) = StableDiffusionFacadeOperations.traceImagePhase(this, phase, detail, throwable)

    internal fun clearImageRequestTrace() =
        StableDiffusionFacadeOperations.clearImageRequestTrace(this)

    internal fun getLastImageRequestTraceForTests(): List<ImageGenerationTraceEvent> =
        StableDiffusionFacadeOperations.getLastImageRequestTraceForTests(this)

    internal fun bitmapToRgbBytesForExecution(bitmap: Bitmap): Triple<ByteArray, Int, Int> =
        StableDiffusionFacadeOperations.bitmapToRgbBytesForExecution(this, bitmap)

    internal fun convertFramesToBitmapsForExecution(
        frameBytesRgb24: Array<ByteArray>,
        width: Int,
        height: Int,
    ): List<Bitmap> =
        StableDiffusionFacadeOperations.convertFramesToBitmapsForExecution(this, frameBytesRgb24, width, height)

    internal fun warnIfLowMemoryForExecution(estimatedAdditionalBytes: Long) =
        StableDiffusionFacadeOperations.warnIfLowMemoryForExecution(this, estimatedAdditionalBytes)

    internal fun estimateFrameFootprintBytesForExecution(width: Int, height: Int, frameCount: Int): Long =
        StableDiffusionFacadeOperations.estimateFrameFootprintBytesForExecution(this, width, height, frameCount)

    internal fun readNativeMemoryMbForExecution(): Long =
        StableDiffusionFacadeOperations.readNativeMemoryMbForExecution(this)

    internal fun nativeIsEasyCacheSupportedForExecution(): Boolean =
        StableDiffusionFacadeOperations.nativeIsEasyCacheSupportedForExecution(this)

    internal fun updateCachedProgressCallback(callback: VideoProgressCallback?) {
        cachedProgressCallback = callback
        runtimeState.cachedProgressCallback = callback
    }

    suspend fun txt2img(params: GenerateParams): Bitmap =
        StableDiffusionFacadeOperations.txt2img(this, params)

    fun isEasyCacheSupported(): Boolean = StableDiffusionFacadeOperations.isEasyCacheSupported(this)

    override fun close() {
        // T096: Proper cleanup - cancel any ongoing generation, destroy native context, reset state
        if (runtimeState.closed.get()) {
            return
        }
        runCatching { cancelGeneration() }
        if (!runtimeState.closed.compareAndSet(false, true)) {
            return
        }
        // If tests have overridden the native bridge, the JNI library may not be loaded
        // so avoid calling nativeDestroy to prevent UnsatisfiedLinkError. See override
        // helpers in the companion object.
        if (!supportNativeBridgeOverriddenForTests() && supportIsNativeLibraryAvailable() && handle != 0L) {
            // Wait for any in-flight generation to observe the cancellation above and
            // release the mutex — destroying the native context while txt2img/txt2vid is
            // still inside the native call would be a use-after-free.
            runBlocking {
                runtimeState.generationMutex.withLock {
                    nativeDestroy(handle)
                }
            }
        }
        runtimeState.cancellationRequested.set(false)
        runtimeState.modelMetadata = null
        runtimeState.easyCacheSupported = null
        runtimeState.lastGenerationMetrics = null
        runtimeState.cachedProgressCallback = null
        cachedProgressCallback = null
        runtimeState.txt2imgPixelBuffer = null
        runtimeState.clearImageTraceState()
    }

    private external fun nativeDestroy(handle: Long)

    internal fun handleForExecution(): Long = handle

    @JvmName("nativeTxt2Img")
    internal external fun nativeTxt2Img(
            handle: Long,
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            steps: Int,
            cfg: Float,
            seed: Long,
            vaeTiling: Boolean,
            easyCacheEnabled: Boolean = false,
            easyCacheReuseThreshold: Float = 0.2f,
            easyCacheStartPercent: Float = 0.15f,
            easyCacheEndPercent: Float = 0.95f,
    ): ByteArray?

    @JvmName("nativeTxt2ImgArgb")
    internal external fun nativeTxt2ImgArgb(
            handle: Long,
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            steps: Int,
            cfg: Float,
            seed: Long,
            vaeTiling: Boolean,
            easyCacheEnabled: Boolean = false,
            easyCacheReuseThreshold: Float = 0.2f,
            easyCacheStartPercent: Float = 0.15f,
            easyCacheEndPercent: Float = 0.95f,
    ): IntArray?

    @JvmName("nativeTxt2Vid")
    internal external fun nativeTxt2Vid(
            handle: Long,
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            videoFrames: Int,
            steps: Int,
            cfg: Float,
            seed: Long,
            sampleMethod: Int,
            scheduler: Int,
            strength: Float,
            initImage: ByteArray?,
            initWidth: Int,
            initHeight: Int,
            vaceStrength: Float,
            easyCacheEnabled: Boolean = false,
            easyCacheReuseThreshold: Float = 0.2f,
            easyCacheStartPercent: Float = 0.15f,
            easyCacheEndPercent: Float = 0.95f,
    ): Array<ByteArray>?

    @JvmName("nativeSetProgressCallback")
    internal external fun nativeSetProgressCallback(
            handle: Long,
            callback: VideoProgressCallback?,
    )

    @JvmName("nativeCancelGeneration")
    internal external fun nativeCancelGeneration(handle: Long)

    @JvmName("nativePrecomputeCondition")
    internal external fun nativePrecomputeCondition(
            handle: Long,
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            clipSkip: Int,
    ): Array<Any?>?

    @JvmName("nativeTxt2VidWithPrecomputedCondition")
    internal external fun nativeTxt2VidWithPrecomputedCondition(
            handle: Long,
            prompt: String,
            negative: String?,
            width: Int,
            height: Int,
            videoFrames: Int,
            steps: Int,
            cfg: Float,
            seed: Long,
            sampleMethod: Int,
            scheduler: Int,
            strength: Float,
            initImage: ByteArray?,
            initWidth: Int,
            initHeight: Int,
            cond: Array<Any?>?,
            uncond: Array<Any?>?,
            vaceStrength: Float,
            easyCacheEnabled: Boolean = false,
            easyCacheReuseThreshold: Float = 0.2f,
            easyCacheStartPercent: Float = 0.15f,
            easyCacheEndPercent: Float = 0.95f,
    ): Array<ByteArray>?

    @JvmName("nativeTxt2ImgWithPrecomputedCondition")
    internal external fun nativeTxt2ImgWithPrecomputedCondition(
            handle: Long,
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            steps: Int,
            cfg: Float,
            seed: Long,
            cond: Array<Any?>?,
            uncond: Array<Any?>?,
            easyCacheEnabled: Boolean = false,
            easyCacheReuseThreshold: Float = 0.2f,
            easyCacheStartPercent: Float = 0.15f,
            easyCacheEndPercent: Float = 0.95f
    ): ByteArray?

    @JvmName("nativeIsEasyCacheSupported")
    internal external fun nativeIsEasyCacheSupported(handle: Long): Boolean

    @JvmName("bitmapToRgbBytes")
    internal fun bitmapToRgbBytes(bitmap: Bitmap): Triple<ByteArray, Int, Int> {
        return StableDiffusionOutputSupport.bitmapToRgbBytes(bitmap, runtimeState.rgbBytesThreadLocal)
    }

    private fun rgbBytesToBitmap(rgb: ByteArray, width: Int, height: Int): Bitmap =
        io.aatricks.llmedge.vision.ImageUtils.rgbBytesToBitmap(rgb, width, height)

    @JvmName("convertFramesToBitmaps")
    internal fun convertFramesToBitmaps(
            frameBytesRgb24: Array<ByteArray>,
            width: Int,
            height: Int,
    ): List<Bitmap> {
        return StableDiffusionOutputSupport.convertFramesToBitmaps(
            frameBytesRgb24 = frameBytesRgb24,
            width = width,
            height = height,
            onRemainingFrames = { remaining ->
                warnIfLowMemory(estimateFrameFootprintBytes(width, height, remaining))
            },
        )
    }

    private fun determineBatchSize(frameCount: Int): Int =
        StableDiffusionOutputSupport.determineBatchSizeForTests(frameCount)

    /**
     * Wrapper that calls the native PrecomputeCondition API and converts to a Kotlin type.
     *
     * This computes a single conditioning for the provided [prompt]. If you intend to use CFG
     * ($\text{cfgScale} \neq 1$), also precompute an unconditional/negative conditioning (e.g. with
     * an empty prompt or your negative prompt) and pass it as `uncond` to
     * [txt2VidWithPrecomputedCondition].
     */
    suspend fun precomputeCondition(
            prompt: String,
            negative: String = "",
            width: Int = 512,
            height: Int = 512,
            clipSkip: Int = -1
    ): PrecomputedCondition? =
        StableDiffusionExecutor.precomputeCondition(this, prompt, negative, width, height, clipSkip)

    /**
     * Variant of txt2vid that accepts precomputed conditioning for both cond/uncond.
     *
     * Note: if [VideoGenerateParams.cfgScale] is not 1.0, you should pass a non-null [uncond]. If
     * [uncond] is null, the native layer will log a warning and run with CFG disabled to avoid
     * crashing.
     */
    suspend fun txt2VidWithPrecomputedCondition(
            params: VideoGenerateParams,
            cond: PrecomputedCondition?,
            uncond: PrecomputedCondition? = null,
            onProgress: VideoProgressCallback? = null,
    ): List<Bitmap> = StableDiffusionExecutor.txt2VidWithPrecomputedCondition(
        this,
        params,
        cond,
        uncond,
        onProgress,
    )

    @JvmName("warnIfLowMemory")
    internal fun warnIfLowMemory(estimatedAdditionalBytes: Long) {
        StableDiffusionOutputSupport.warnIfLowMemory(
            logTag = "StableDiffusion",
            estimatedAdditionalBytes = estimatedAdditionalBytes,
            bytesInMb = BYTES_IN_MB,
            memoryPressureThreshold = MEMORY_PRESSURE_THRESHOLD,
        )
    }

    @JvmName("estimateFrameFootprintBytes")
    internal fun estimateFrameFootprintBytes(width: Int, height: Int, frameCount: Int): Long =
            StableDiffusionOutputSupport.estimateFrameFootprintBytes(width, height, frameCount)

    @JvmName("readNativeMemoryMb")
    internal fun readNativeMemoryMb(): Long =
            StableDiffusionOutputSupport.readNativeMemoryMb(BYTES_IN_MB)
}
