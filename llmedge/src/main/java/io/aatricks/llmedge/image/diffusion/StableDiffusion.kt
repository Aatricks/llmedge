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
import io.aatricks.llmedge.core.NativeBridgeProvider
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeLibraryLoader
import io.aatricks.llmedge.core.UnsupportedModelException
import io.aatricks.llmedge.image.diffusion.internal.StableDiffusionExecutor
import io.aatricks.llmedge.image.diffusion.internal.StableDiffusionLoader
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.Executors
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal fun interface StableDiffusionNativeLibrarySupport {
    fun ensureLoaded()
}

class StableDiffusion internal constructor(
    private val handle: Long,
    private val nativeLibrarySupport: StableDiffusionNativeLibrarySupport = currentNativeLibrarySupport(),
) : AutoCloseable {
    // Preserve the legacy reflective test seam while routing through explicit native loading.
    private constructor(handle: Long) : this(handle, currentNativeLibrarySupport())

    init {
        nativeLibrarySupport.ensureLoaded()
    }

    private val runtimeState = StableDiffusionState(handle)
    private val nativeBridge: NativeBridge = Companion.nativeBridgeProvider.create(this)
    // Legacy reflective tests still reach into these fields directly, so keep thin mirrors.
    private val cancellationRequested = runtimeState.cancellationRequested
    private var cachedProgressCallback: VideoProgressCallback? = null

    internal interface NativeBridge {
        fun txt2img(
                handle: Long,
                prompt: String,
                negative: String,
                width: Int,
                height: Int,
                steps: Int,
                cfg: Float,
                seed: Long,
                vaeTiling: Boolean,
                easyCacheEnabled: Boolean,
                easyCacheReuseThreshold: Float,
                easyCacheStartPercent: Float,
                easyCacheEndPercent: Float,
        ): ByteArray?

        /** Fast path: returns ARGB_8888 pixels directly, avoiding Kotlin RGB→ARGB conversion */
        fun txt2imgArgb(
                handle: Long,
                prompt: String,
                negative: String,
                width: Int,
                height: Int,
                steps: Int,
                cfg: Float,
                seed: Long,
                vaeTiling: Boolean,
                easyCacheEnabled: Boolean,
                easyCacheReuseThreshold: Float,
                easyCacheStartPercent: Float,
                easyCacheEndPercent: Float,
        ): IntArray? = null

        fun txt2vid(
                handle: Long,
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
                vaceStrength: Float,
                easyCacheEnabled: Boolean,
                easyCacheReuseThreshold: Float,
                easyCacheStartPercent: Float,
                easyCacheEndPercent: Float,
        ): Array<ByteArray>?

        fun precomputeCondition(
                handle: Long,
                prompt: String,
                negative: String,
                width: Int,
                height: Int,
                clipSkip: Int
        ): PrecomputedCondition? = null

        fun txt2vidWithPrecomputedCondition(
                handle: Long,
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
                cond: PrecomputedCondition?,
                uncond: PrecomputedCondition?,
                vaceStrength: Float,
                easyCacheEnabled: Boolean,
                easyCacheReuseThreshold: Float,
                easyCacheStartPercent: Float,
                easyCacheEndPercent: Float,
        ): Array<ByteArray>? =
                txt2vid(
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

        fun setProgressCallback(handle: Long, callback: VideoProgressCallback?)
        fun cancelGeneration(handle: Long)

        fun txt2ImgWithPrecomputedCondition(
                handle: Long,
                prompt: String,
                negative: String,
                width: Int,
                height: Int,
                steps: Int,
                cfg: Float,
                seed: Long,
                cond: PrecomputedCondition?,
                uncond: PrecomputedCondition?,
                easyCacheEnabled: Boolean,
                easyCacheReuseThreshold: Float,
                easyCacheStartPercent: Float,
                easyCacheEndPercent: Float,
        ): ByteArray? =
                txt2img(
                        handle,
                        prompt,
                        negative,
                        width,
                        height,
                        steps,
                        cfg,
                        seed,
                        true,
                        easyCacheEnabled,
                        easyCacheReuseThreshold,
                        easyCacheStartPercent,
                        easyCacheEndPercent
                )
    }

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
        private const val LOG_TAG = "StableDiffusion"
        private const val BYTES_IN_MB = 1024L * 1024L
        private const val MEMORY_PRESSURE_THRESHOLD = 0.85f
        private const val DIFFUSION_DISPATCHER_THREADS = 2

        /**
         * Dedicated dispatcher for diffusion workloads. Use a small fixed pool instead of a
         * single thread so one wedged native generation cannot block every later image/video
         * request in the process. Per-model generation is still serialized separately by mutexes.
         */
        private val diffusionWorkerIds = AtomicInteger(0)

        val diffusionDispatcher: CoroutineDispatcher =
            Executors.newFixedThreadPool(DIFFUSION_DISPATCHER_THREADS) {
                val workerId = diffusionWorkerIds.incrementAndGet()
                Thread(it, "llmedge-diffusion-$workerId").apply { isDaemon = true }
            }.asCoroutineDispatcher()

        // Kept as a reflective seam for heuristic tests that validate the combined load policy.
        @JvmStatic
        private fun computeEffectiveSequentialLoad(
            context: Context,
            resolvedModelPath: String,
            sequentialLoad: Boolean?,
            preferPerformanceMode: Boolean,
            activityManagerOverride: android.app.ActivityManager?,
        ): Pair<Boolean, Long> =
            StableDiffusionLoadHeuristics.computeEffectiveSequentialLoad(
                context = context,
                resolvedModelPath = resolvedModelPath,
                sequentialLoad = sequentialLoad,
                preferPerformanceMode = preferPerformanceMode,
                activityManagerOverride = activityManagerOverride,
            )

        private fun logD(tag: String, message: String) = AndroidLogAdapter.d(tag, message)

        private fun logI(tag: String, message: String) = AndroidLogAdapter.i(tag, message)

        private fun logW(tag: String, message: String) = AndroidLogAdapter.w(tag, message)

        private fun logE(tag: String, message: String, throwable: Throwable? = null) =
            AndroidLogAdapter.e(tag, message, throwable)

        @Volatile private var isNativeLibraryAvailable: Boolean = false
        // Flag set by tests when overriding the native bridge to a test mock so we avoid
        // calling actual JNI functions like nativeDestroy during Android instrumentation tests.
        private var nativeBridgeOverriddenForTests: Boolean = false
        private val defaultNativeLibrarySupport =
            StableDiffusionNativeLibrarySupport {
                val disableNativeLoad = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")
                isNativeLibraryAvailable = !disableNativeLoad
                if (disableNativeLoad) {
                    logI(LOG_TAG, "Native load disabled via llmedge.disableNativeLoad=true")
                } else {
                    NativeLibraryLoader.ensureStableDiffusionLoaded(
                        required = true,
                        onDebug = { message -> logD(LOG_TAG, message) },
                        onError = { message, throwable -> logE(LOG_TAG, message, throwable) },
                        verifyBindings = ::nativeCheckBindings,
                    )
                }
            }
        private val noOpNativeLibrarySupport = StableDiffusionNativeLibrarySupport { }

        @Volatile
        private var nativeLibrarySupportOverride: StableDiffusionNativeLibrarySupport? = null

        private val defaultNativeBridgeProvider: (StableDiffusion) -> NativeBridge =
            StableDiffusionNativeBridgeSupport.defaultProvider()

        private val nativeBridgeProvider = NativeBridgeProvider(defaultNativeBridgeProvider)

        internal fun currentNativeLibrarySupport(): StableDiffusionNativeLibrarySupport =
            nativeLibrarySupportOverride ?: defaultNativeLibrarySupport

        /**
         * Helper for tests and runtime checks to verify whether the native sdcpp library is
         * implemented and correctly linked. This attempts to call the JNI `nativeCheckBindings`
         * method and returns false if the library is not present.
         */
        @JvmStatic
        fun isNativeLibraryLoaded(): Boolean {
            return try {
                nativeCheckBindings()
            } catch (t: Throwable) {
                false
            }
        }

        internal fun enableNativeBridgeForTests() {
            if (!isNativeLibraryAvailable) {
                isNativeLibraryAvailable = true
            }
        }

        internal fun overrideNativeBridgeForTests(provider: (StableDiffusion) -> NativeBridge) {
            nativeBridgeProvider.override(provider)
            nativeBridgeOverriddenForTests = true
            nativeLibrarySupportOverride = noOpNativeLibrarySupport
        }

        internal fun resetNativeBridgeForTests() {
            nativeBridgeProvider.reset()
            nativeBridgeOverriddenForTests = false
            nativeLibrarySupportOverride = null
        }

        /**
         * Get the number of Vulkan devices available on this system
         * @return Number of Vulkan-capable devices, or 0 if Vulkan is not available
         */
        @JvmStatic
        fun getVulkanDeviceCount(): Int {
            return try {
                currentNativeLibrarySupport().ensureLoaded()
                nativeGetVulkanDeviceCount()
            } catch (e: Throwable) {
                0
            }
        }

        /**
         * Get Vulkan device memory information
         * @param deviceIndex Index of the device to query (default 0)
         * @return LongArray with [freeMemory, totalMemory] in bytes, or null if unavailable
         */
        @JvmStatic
        fun getVulkanDeviceMemory(deviceIndex: Int = 0): LongArray? {
            return try {
                currentNativeLibrarySupport().ensureLoaded()
                nativeGetVulkanDeviceMemory(deviceIndex)
            } catch (e: Throwable) {
                null
            }
        }

        /**
         * Get a human-readable Vulkan device description.
         */
        @JvmStatic
        fun getVulkanDeviceDescription(deviceIndex: Int = 0): String? {
            return try {
                currentNativeLibrarySupport().ensureLoaded()
                nativeGetVulkanDeviceDescription(deviceIndex)
            } catch (_: Throwable) {
                null
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
            return try {
                currentNativeLibrarySupport().ensureLoaded()
                nativeEstimateModelParamsMemory(modelPath, deviceIndex)
            } catch (t: Throwable) {
                0L
            }
        }

        @JvmStatic
        fun checkBindings(): Boolean {
            return try {
                currentNativeLibrarySupport().ensureLoaded()
                nativeCheckBindings()
            } catch (t: Throwable) {
                false
            }
        }

        @JvmStatic
        fun isOpenClAvailable(): Boolean {
            return try {
                currentNativeLibrarySupport().ensureLoaded()
                nativeIsOpenClAvailable()
            } catch (_: Throwable) {
                false
            }
        }

        internal fun supportNativeCreate(
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

        @JvmStatic
        private external fun nativeCreate(
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

        private suspend fun inferVideoModelMetadata(
                resolvedModelPath: String,
                modelId: String?,
                explicitFilename: String?,
        ): VideoModelMetadata =
            StableDiffusionMetadataSupport.inferVideoModelMetadata(
                resolvedModelPath = resolvedModelPath,
                modelId = modelId,
                explicitFilename = explicitFilename,
            )

        private fun validateResolvedAssets(
            modelPath: String,
            vaePath: String?,
            t5xxlPath: String?,
            taesdPath: String?,
            loraModelDir: String?,
        ) {
            StableDiffusionLoadHeuristics.validateResolvedAssets(
                modelPath = modelPath,
                vaePath = vaePath,
                t5xxlPath = t5xxlPath,
                taesdPath = taesdPath,
                loraModelDir = loraModelDir,
            )
        }

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

        internal fun supportLogWarning(message: String) = logW(LOG_TAG, message)

        internal fun supportIsNativeLibraryAvailable(): Boolean = isNativeLibraryAvailable

        internal fun supportNativeBridgeOverriddenForTests(): Boolean = nativeBridgeOverriddenForTests

    }

    internal fun updateModelMetadata(metadata: VideoModelMetadata?) {
        runtimeState.modelMetadata = metadata
        runtimeState.easyCacheSupported =
            if (!Companion.isNativeLibraryAvailable || Companion.nativeBridgeOverriddenForTests) {
                metadata?.let(StableDiffusionMetadataSupport::supportsEasyCache)
            } else {
                null
            }
    }

    fun isVideoModel(): Boolean {
        val metadata = runtimeState.modelMetadata ?: return false
        return StableDiffusionMetadataSupport.isVideoModel(metadata)
    }

    suspend fun txt2vid(
            params: VideoGenerateParams,
            onProgress: VideoProgressCallback? = null,
        ): List<Bitmap> = StableDiffusionExecutor.txt2vid(this, params, onProgress)

    fun setProgressCallback(callback: VideoProgressCallback?) {
        StableDiffusionExecutor.setProgressCallback(this, callback)
    }

    fun cancelGeneration() {
        StableDiffusionExecutor.cancelGeneration(this)
    }

    fun getLastGenerationMetrics(): GenerationMetrics? = runtimeState.lastGenerationMetrics

    internal val bridge: NativeBridge
        get() = nativeBridge

    internal val state: StableDiffusionState
        get() = runtimeState

    internal fun beginImageRequestTrace(requestId: Long) {
        runtimeState.beginImageTrace(requestId)
    }

    internal fun traceImagePhase(
        phase: ImageGenerationPhase,
        detail: String? = null,
        throwable: Throwable? = null,
    ) {
        if (phase.isTerminal() && runtimeState.currentImagePhase?.isTerminal() == true) {
            return
        }
        val requestId = runtimeState.appendImageTrace(phase, detail) ?: return
        val message =
            buildString {
                append("requestId=")
                append(requestId)
                append(", phase=")
                append(phase.name)
                if (!detail.isNullOrBlank()) {
                    append(", detail=")
                    append(detail)
                }
            }
        if (throwable == null) {
            AndroidLogAdapter.i("StableDiffusionTrace", message)
        } else {
            AndroidLogAdapter.e("StableDiffusionTrace", message, throwable)
        }
    }

    internal fun clearImageRequestTrace() {
        runtimeState.clearImageTraceState()
    }

    internal fun getLastImageRequestTraceForTests(): List<ImageGenerationTraceEvent> =
        runtimeState.snapshotLastImageTrace()

    internal fun bitmapToRgbBytesForExecution(bitmap: Bitmap): Triple<ByteArray, Int, Int> =
        bitmapToRgbBytes(bitmap)

    internal fun convertFramesToBitmapsForExecution(
        frameBytesRgb24: Array<ByteArray>,
        width: Int,
        height: Int,
    ): List<Bitmap> = convertFramesToBitmaps(frameBytesRgb24, width, height)

    internal fun warnIfLowMemoryForExecution(estimatedAdditionalBytes: Long) =
        warnIfLowMemory(estimatedAdditionalBytes)

    internal fun estimateFrameFootprintBytesForExecution(width: Int, height: Int, frameCount: Int): Long =
        estimateFrameFootprintBytes(width, height, frameCount)

    internal fun readNativeMemoryMbForExecution(): Long = readNativeMemoryMb()

    internal fun nativeIsEasyCacheSupportedForExecution(): Boolean =
        nativeIsEasyCacheSupported(handle)

    internal fun updateCachedProgressCallback(callback: VideoProgressCallback?) {
        cachedProgressCallback = callback
        runtimeState.cachedProgressCallback = callback
    }

    suspend fun txt2img(params: GenerateParams): Bitmap {
        traceImagePhase(
            ImageGenerationPhase.TXT2IMG_ENTER,
            "StableDiffusion.txt2img entered width=${params.width} height=${params.height} steps=${params.steps}",
        )
        return StableDiffusionExecutor.txt2img(this, params)
    }

    fun isEasyCacheSupported(): Boolean {
        return StableDiffusionExecutor.isEasyCacheSupported(this)
    }

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
        if (!Companion.nativeBridgeOverriddenForTests && isNativeLibraryAvailable && handle != 0L) {
            nativeDestroy(handle)
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

    internal external fun nativeSetProgressCallback(
            handle: Long,
            callback: VideoProgressCallback?,
    )

    internal external fun nativeCancelGeneration(handle: Long)

    internal external fun nativePrecomputeCondition(
            handle: Long,
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            clipSkip: Int,
    ): Array<Any?>?

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

    internal external fun nativeIsEasyCacheSupported(handle: Long): Boolean

    private fun bitmapToRgbBytes(bitmap: Bitmap): Triple<ByteArray, Int, Int> {
        return StableDiffusionOutputSupport.bitmapToRgbBytes(bitmap, runtimeState.rgbBytesThreadLocal)
    }

    private fun rgbBytesToBitmap(rgb: ByteArray, width: Int, height: Int): Bitmap =
        io.aatricks.llmedge.vision.ImageUtils.rgbBytesToBitmap(rgb, width, height)

    private fun convertFramesToBitmaps(
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

    private fun warnIfLowMemory(estimatedAdditionalBytes: Long) {
        StableDiffusionOutputSupport.warnIfLowMemory(
            logTag = LOG_TAG,
            estimatedAdditionalBytes = estimatedAdditionalBytes,
            bytesInMb = BYTES_IN_MB,
            memoryPressureThreshold = MEMORY_PRESSURE_THRESHOLD,
        )
    }

    private fun estimateFrameFootprintBytes(width: Int, height: Int, frameCount: Int): Long =
            StableDiffusionOutputSupport.estimateFrameFootprintBytes(width, height, frameCount)

    private fun readNativeMemoryMb(): Long =
            StableDiffusionOutputSupport.readNativeMemoryMb(BYTES_IN_MB)
}
