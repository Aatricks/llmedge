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
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.Executors
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class StableDiffusion internal constructor(private val handle: Long) : AutoCloseable {
    // Serialize concurrent generation calls - native library is not guaranteed to be reentrant.
    private val generationMutex = Mutex()
    private var modelMetadata: VideoModelMetadata? = null
    private var easyCacheSupported: Boolean? = null
    private val cancellationRequested = AtomicBoolean(false)
    private val rgbBytesThreadLocal = ThreadLocal<ByteArray>()
    // Reusable pixel buffer for txt2img RGB→ARGB conversion
    private var txt2imgPixelBuffer: IntArray? = null

    private var vulkanEnabledForMetrics: Boolean = false

    @Volatile private var cachedProgressCallback: VideoProgressCallback? = null

    @Volatile private var lastGenerationMetrics: GenerationMetrics? = null
    private val nativeBridge: NativeBridge = Companion.nativeBridgeProvider.create(this)

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

        /**
         * Dedicated single-thread dispatcher for diffusion workloads. Keeps heavy generation
         * tasks off the shared IO pool so they don't starve network/disk operations.
         */
        val diffusionDispatcher: CoroutineDispatcher =
            Executors.newSingleThreadExecutor { r ->
                Thread(r, "llmedge-diffusion").apply { isDaemon = true }
            }.asCoroutineDispatcher()

        private fun logD(tag: String, message: String) = AndroidLogAdapter.d(tag, message)

        private fun logI(tag: String, message: String) = AndroidLogAdapter.i(tag, message)

        private fun logW(tag: String, message: String) = AndroidLogAdapter.w(tag, message)

        private fun logE(tag: String, message: String, throwable: Throwable? = null) =
            AndroidLogAdapter.e(tag, message, throwable)

        // Dummy instance used to invoke static native methods that are now at the class level.
        private val staticInvoker: StableDiffusion by lazy { StableDiffusion(0L) }

        @Volatile private var isNativeLibraryAvailable: Boolean
        // Flag set by tests when overriding the native bridge to a test mock so we avoid
        // calling actual JNI functions like nativeDestroy during Android instrumentation tests.
        private var nativeBridgeOverriddenForTests: Boolean = false

        private val defaultNativeBridgeProvider: (StableDiffusion) -> NativeBridge = { instance ->
            object : NativeBridge {
                override fun txt2img(
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
                ): ByteArray? =
                        instance.nativeTxt2Img(
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

                override fun txt2imgArgb(
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
                ): IntArray? =
                        instance.nativeTxt2ImgArgb(
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

                override fun txt2vid(
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
                ): Array<ByteArray>? =
                        instance.nativeTxt2Vid(
                                handle,
                                prompt,
                                negative,
                                width,
                                height,
                                videoFrames,
                                steps,
                                cfg,
                                seed,
                                sampleMethod.id,
                                scheduler.id,
                                strength,
                                initImage = initImage,
                                initWidth = initWidth,
                                initHeight = initHeight,
                                vaceStrength = vaceStrength,
                                easyCacheEnabled = easyCacheEnabled,
                                easyCacheReuseThreshold = easyCacheReuseThreshold,
                                easyCacheStartPercent = easyCacheStartPercent,
                                easyCacheEndPercent = easyCacheEndPercent
                        )

                override fun precomputeCondition(
                        handle: Long,
                        prompt: String,
                        negative: String,
                        width: Int,
                        height: Int,
                        clipSkip: Int
                ): PrecomputedCondition? {
                    val raw =
                            instance.nativePrecomputeCondition(
                                    handle,
                                    prompt,
                                    negative,
                                    width,
                                    height,
                                    clipSkip
                            )
                                    ?: return null
                                return StableDiffusionConditionInterop.fromNativeRaw(raw)
                }

                override fun txt2vidWithPrecomputedCondition(
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
                        easyCacheEndPercent: Float
                ): Array<ByteArray>? {
                    return instance.nativeTxt2VidWithPrecomputedCondition(
                            handle,
                            prompt,
                            negative,
                            width,
                            height,
                            videoFrames,
                            steps,
                            cfg,
                            seed,
                            sampleMethod.id,
                            scheduler.id,
                            strength,
                            initImage = initImage,
                            initWidth = initWidth,
                            initHeight = initHeight,
                            cond = StableDiffusionConditionInterop.toNativeArray(cond),
                            uncond = StableDiffusionConditionInterop.toNativeArray(uncond),
                            vaceStrength = vaceStrength,
                            easyCacheEnabled = easyCacheEnabled,
                            easyCacheReuseThreshold = easyCacheReuseThreshold,
                            easyCacheStartPercent = easyCacheStartPercent,
                            easyCacheEndPercent = easyCacheEndPercent,
                    )
                }

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) {
                    instance.nativeSetProgressCallback(handle, callback)
                }

                override fun cancelGeneration(handle: Long) {
                    instance.nativeCancelGeneration(handle)
                }

                override fun txt2ImgWithPrecomputedCondition(
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
                        easyCacheEndPercent: Float
                ): ByteArray? {
                    return instance.nativeTxt2ImgWithPrecomputedCondition(
                            handle,
                            prompt,
                            negative,
                            width,
                            height,
                            steps,
                            cfg,
                            seed,
                            StableDiffusionConditionInterop.toNativeArray(cond),
                            StableDiffusionConditionInterop.toNativeArray(uncond),
                            easyCacheEnabled,
                            easyCacheReuseThreshold,
                            easyCacheStartPercent,
                            easyCacheEndPercent
                    )
                }
            }
        }

        private val nativeBridgeProvider = NativeBridgeProvider(defaultNativeBridgeProvider)

        init {
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
        }

        internal fun resetNativeBridgeForTests() {
            nativeBridgeProvider.reset()
            nativeBridgeOverriddenForTests = false
        }

        /**
         * Get the number of Vulkan devices available on this system
         * @return Number of Vulkan-capable devices, or 0 if Vulkan is not available
         */
        @JvmStatic
        fun getVulkanDeviceCount(): Int {
            return try {
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
                nativeEstimateModelParamsMemory(modelPath, deviceIndex)
            } catch (t: Throwable) {
                0L
            }
        }

        @JvmStatic
        fun checkBindings(): Boolean {
            return try {
                nativeCheckBindings()
            } catch (t: Throwable) {
                false
            }
        }

        @JvmStatic
        fun isOpenClAvailable(): Boolean {
            return try {
                nativeIsOpenClAvailable()
            } catch (_: Throwable) {
                false
            }
        }

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

        internal suspend fun supportInferVideoModelMetadata(
            resolvedModelPath: String,
            modelId: String?,
            explicitFilename: String?,
        ): VideoModelMetadata =
            inferVideoModelMetadata(resolvedModelPath, modelId, explicitFilename)

        internal fun supportValidateResolvedAssets(
            modelPath: String,
            vaePath: String?,
            t5xxlPath: String?,
            taesdPath: String?,
            loraModelDir: String?,
        ) = validateResolvedAssets(modelPath, vaePath, t5xxlPath, taesdPath, loraModelDir)

        internal fun supportLogLoadFallback(message: String) = logW(LOG_TAG, message)

        internal fun supportLogWarning(message: String) = logW(LOG_TAG, message)

        internal fun supportIsNativeLibraryAvailable(): Boolean = isNativeLibraryAvailable

        internal fun supportNativeBridgeOverriddenForTests(): Boolean = nativeBridgeOverriddenForTests

        internal fun supportCreateLoadedInstance(
            context: Context,
            resolved: StableDiffusionResolvedAssets,
            taesdPath: String?,
            nThreads: Int,
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
            flowShift: Float,
            loraModelDir: String?,
            loraApplyMode: LoraApplyMode,
            preferredBackend: ComputeBackend?,
            allowBackendFallbackToCpu: Boolean,
        ): StableDiffusion =
            createLoadedInstance(
                context = context,
                resolved = resolved,
                taesdPath = taesdPath,
                nThreads = nThreads,
                offloadToCpu = offloadToCpu,
                keepClipOnCpu = keepClipOnCpu,
                keepVaeOnCpu = keepVaeOnCpu,
                flashAttn = flashAttn,
                vaeDecodeOnly = vaeDecodeOnly,
                sequentialLoad = sequentialLoad,
                allowOpenCl = allowOpenCl,
                allowVulkan = allowVulkan,
                forceVulkan = forceVulkan,
                preferPerformanceMode = preferPerformanceMode,
                flowShift = flowShift,
                loraModelDir = loraModelDir,
                loraApplyMode = loraApplyMode,
                preferredBackend = preferredBackend,
                allowBackendFallbackToCpu = allowBackendFallbackToCpu,
            )

        private fun createLoadedInstance(
            context: Context,
            resolved: StableDiffusionResolvedAssets,
            taesdPath: String?,
            nThreads: Int,
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
            flowShift: Float,
            loraModelDir: String?,
            loraApplyMode: LoraApplyMode,
            preferredBackend: ComputeBackend?,
            allowBackendFallbackToCpu: Boolean,
        ): StableDiffusion {
            val loadPlan =
                StableDiffusionLoadHeuristics.planLoad(
                    context = context,
                    resolvedModelPath = resolved.modelPath,
                    sequentialLoad = sequentialLoad,
                    preferPerformanceMode = preferPerformanceMode,
                    offloadToCpu = offloadToCpu,
                    keepClipOnCpu = keepClipOnCpu,
                    keepVaeOnCpu = keepVaeOnCpu,
                    allowOpenCl = allowOpenCl,
                    allowVulkan = allowVulkan,
                    forceVulkan = forceVulkan,
                )
            StableDiffusionLoadHeuristics.warnIfLargeModelOnLowRam(
                metadata = resolved.metadata,
                memorySnapshot = loadPlan.memorySnapshot,
            ) { message -> logW(LOG_TAG, message) }

            logLoadPlan(
                resolvedModelPath = resolved.modelPath,
                nThreads = nThreads,
                loadPlan = loadPlan,
                flashAttn = flashAttn,
            )

            val requestedVulkan = loadPlan.chosenBackend == ComputeBackend.VULKAN

            val handle =
                createHandleWithBackendFallback(
                    resolved = resolved,
                    taesdPath = taesdPath,
                    nThreads = nThreads,
                    loadPlan = loadPlan,
                    flashAttn = flashAttn,
                    vaeDecodeOnly = vaeDecodeOnly,
                    flowShift = flowShift,
                    loraModelDir = loraModelDir,
                    loraApplyMode = loraApplyMode,
                    allowBackendFallbackToCpu = allowBackendFallbackToCpu,
                )
            if (handle == 0L) {
                throw ModelLoadException(
                    resolved.modelPath,
                    createLoadFailureMessage(
                        resolvedModelPath = resolved.modelPath,
                        taesdPath = taesdPath,
                        resolvedVaePath = resolved.vaePath,
                    ),
                )
            }

            val instance = StableDiffusion(handle)
            instance.vulkanEnabledForMetrics = requestedVulkan
            instance.updateModelMetadata(resolved.metadata)

            if (instance.modelMetadata?.mobileSupported == false) {
                instance.close()
                val paramCount = instance.modelMetadata?.parameterCount ?: "14B"
                throw UnsupportedModelException(
                    "$paramCount models are not supported on mobile devices. " +
                        "Please use 1.3B or 5B model variants instead. " +
                        "14B models require 20-40GB RAM and are designed for desktop/server use only.",
                )
            }

            return instance
        }

        private fun logLoadPlan(
            resolvedModelPath: String,
            nThreads: Int,
            loadPlan: StableDiffusionLoadHeuristics.LoadPlan,
            flashAttn: Boolean,
        ) {
            logI(
                LOG_TAG,
                "Initializing StableDiffusion (effective): modelPath=$resolvedModelPath, " +
                    "nThreads=$nThreads, sequentialLoad=${loadPlan.effectiveSequentialLoad}, " +
                    "offloadToCpu=${loadPlan.effectiveOffloadToCpu}, " +
                    "keepClipOnCpu=${loadPlan.effectiveKeepClipOnCpu}, backend=${loadPlan.chosenBackend}, " +
                    "keepVaeOnCpu=${loadPlan.effectiveKeepVaeOnCpu}, flashAttn=$flashAttn",
            )
            if (loadPlan.chosenDevice >= 0) {
                logI(
                    LOG_TAG,
                    "Vulkan chosenDevice=${loadPlan.chosenDevice}, estimatedModelParamsMB=${String.format("%.2f", loadPlan.estimatedDeviceParamsBytes / 1024.0 / 1024.0)}, freeMB=${String.format("%.2f", loadPlan.freeVulkanBytes / 1024.0 / 1024.0)}",
                )
            }
        }

        private fun createHandleWithBackendFallback(
            resolved: StableDiffusionResolvedAssets,
            taesdPath: String?,
            nThreads: Int,
            loadPlan: StableDiffusionLoadHeuristics.LoadPlan,
            flashAttn: Boolean,
            vaeDecodeOnly: Boolean,
            flowShift: Float,
            loraModelDir: String?,
            loraApplyMode: LoraApplyMode,
            allowBackendFallbackToCpu: Boolean,
        ): Long {
            var effectiveOffloadToCpu = loadPlan.effectiveOffloadToCpu
            var effectiveKeepClipOnCpu = loadPlan.effectiveKeepClipOnCpu
            var effectiveKeepVaeOnCpu = loadPlan.effectiveKeepVaeOnCpu

            val enableOpenCl = loadPlan.chosenBackend == ComputeBackend.OPENCL
            val shouldUseVulkan = loadPlan.chosenBackend == ComputeBackend.VULKAN

            var handle =
                nativeCreateOrThrow(
                    modelPath = resolved.modelPath,
                    vaePath = resolved.vaePath,
                    t5xxlPath = resolved.t5xxlPath,
                    taesdPath = taesdPath,
                    nThreads = nThreads,
                    enableOpenCl = enableOpenCl,
                    useVulkan = shouldUseVulkan,
                    offloadToCpu = effectiveOffloadToCpu,
                    keepClipOnCpu = effectiveKeepClipOnCpu,
                    keepVaeOnCpu = effectiveKeepVaeOnCpu,
                    flashAttn = flashAttn,
                    vaeDecodeOnly = vaeDecodeOnly,
                    flowShift = flowShift,
                    loraModelDir = loraModelDir,
                    loraApplyMode = loraApplyMode,
                )
            if (handle == 0L && allowBackendFallbackToCpu && loadPlan.chosenBackend != ComputeBackend.CPU) {
                logW(LOG_TAG, "nativeCreate failed on ${loadPlan.chosenBackend}; retrying with CPU backend")
                handle =
                    nativeCreateOrThrow(
                        modelPath = resolved.modelPath,
                        vaePath = resolved.vaePath,
                        t5xxlPath = resolved.t5xxlPath,
                        taesdPath = taesdPath,
                        nThreads = nThreads,
                        enableOpenCl = false,
                        useVulkan = false,
                        offloadToCpu = effectiveOffloadToCpu,
                        keepClipOnCpu = effectiveKeepClipOnCpu,
                        keepVaeOnCpu = effectiveKeepVaeOnCpu,
                        flashAttn = flashAttn,
                        vaeDecodeOnly = vaeDecodeOnly,
                        flowShift = flowShift,
                        loraModelDir = loraModelDir,
                        loraApplyMode = loraApplyMode,
                    )
            }
            if (handle == 0L && !effectiveOffloadToCpu) {
                logW(LOG_TAG, "nativeCreate failed on CPU backend; retrying with CPU offload")
                effectiveOffloadToCpu = true
                effectiveKeepClipOnCpu = true
                effectiveKeepVaeOnCpu = true
                handle =
                    nativeCreateOrThrow(
                        modelPath = resolved.modelPath,
                        vaePath = resolved.vaePath,
                        t5xxlPath = resolved.t5xxlPath,
                        taesdPath = taesdPath,
                        nThreads = nThreads,
                        enableOpenCl = false,
                        useVulkan = false,
                        offloadToCpu = effectiveOffloadToCpu,
                        keepClipOnCpu = effectiveKeepClipOnCpu,
                        keepVaeOnCpu = effectiveKeepVaeOnCpu,
                        flashAttn = flashAttn,
                        vaeDecodeOnly = vaeDecodeOnly,
                        flowShift = flowShift,
                        loraModelDir = loraModelDir,
                        loraApplyMode = loraApplyMode,
                    )
            }
            return handle
        }

        private fun nativeCreateOrThrow(
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
            loraApplyMode: LoraApplyMode,
        ): Long =
            try {
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
                    loraApplyMode.id,
                )
            } catch (e: UnsatisfiedLinkError) {
                throw NativeBindingException(
                    libraryName = "sdcpp",
                    detail = "Stable Diffusion JNI bindings are unavailable.",
                    cause = e,
                )
            }

        private fun createLoadFailureMessage(
            resolvedModelPath: String,
            taesdPath: String?,
            resolvedVaePath: String?,
        ): String =
            buildString {
                append("Failed to initialize Stable Diffusion context for $resolvedModelPath.")
                if (taesdPath != null) append(" Custom TAE/TAEHV: $taesdPath.")
                if (resolvedVaePath != null) append(" Custom VAE: $resolvedVaePath.")
                append(" This often happens due to incompatible VAE/TAE weights or insufficient memory. Check logcat for [SmolSD] errors.")
            }
    }

    // Legacy alias for backward compatibility
    @Deprecated("Use SampleMethod enum instead", ReplaceWith("SampleMethod"))
    val EULER_A = SampleMethod.EULER_A
    @Deprecated("Use SampleMethod enum instead", ReplaceWith("SampleMethod"))
    val DDIM = SampleMethod.DDIM_TRAILING
    @Deprecated("Use SampleMethod enum instead", ReplaceWith("SampleMethod"))
    val LCM = SampleMethod.LCM

    internal fun updateModelMetadata(metadata: VideoModelMetadata?) {
        modelMetadata = metadata
        easyCacheSupported =
            if (!Companion.isNativeLibraryAvailable || Companion.nativeBridgeOverriddenForTests) {
                metadata?.let(StableDiffusionMetadataSupport::supportsEasyCache)
            } else {
                null
            }
    }

    fun isVideoModel(): Boolean {
        val metadata = modelMetadata ?: return false
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

    fun getLastGenerationMetrics(): GenerationMetrics? = lastGenerationMetrics

    internal val supportHandle: Long
        get() = handle

    internal val supportNativeBridge: NativeBridge
        get() = nativeBridge

    internal val supportGenerationMutex: Mutex
        get() = generationMutex

    internal val supportCancellationRequested: AtomicBoolean
        get() = cancellationRequested

    internal val supportModelMetadata: VideoModelMetadata?
        get() = modelMetadata

    internal var supportEasyCacheSupported: Boolean?
        get() = easyCacheSupported
        set(value) {
            easyCacheSupported = value
        }

    internal var supportCachedProgressCallback: VideoProgressCallback?
        get() = cachedProgressCallback
        set(value) {
            cachedProgressCallback = value
        }

    internal var supportLastGenerationMetrics: GenerationMetrics?
        get() = lastGenerationMetrics
        set(value) {
            lastGenerationMetrics = value
        }

    internal var supportTxt2imgPixelBuffer: IntArray?
        get() = txt2imgPixelBuffer
        set(value) {
            txt2imgPixelBuffer = value
        }

    internal val supportVulkanEnabledForMetrics: Boolean
        get() = vulkanEnabledForMetrics

    internal fun supportBitmapToRgbBytes(bitmap: Bitmap): Triple<ByteArray, Int, Int> =
        bitmapToRgbBytes(bitmap)

    internal fun supportConvertFramesToBitmaps(
        frameBytesRgb24: Array<ByteArray>,
        width: Int,
        height: Int,
    ): List<Bitmap> = convertFramesToBitmaps(frameBytesRgb24, width, height)

    internal fun supportWarnIfLowMemory(estimatedAdditionalBytes: Long) =
        warnIfLowMemory(estimatedAdditionalBytes)

    internal fun supportEstimateFrameFootprintBytes(width: Int, height: Int, frameCount: Int): Long =
        estimateFrameFootprintBytes(width, height, frameCount)

    internal fun supportReadNativeMemoryMb(): Long = readNativeMemoryMb()

    internal fun supportNativeIsEasyCacheSupported(): Boolean = nativeIsEasyCacheSupported(handle)

    suspend fun txt2img(params: GenerateParams): Bitmap =
            StableDiffusionExecutor.txt2img(this, params)

    fun isEasyCacheSupported(): Boolean {
        return StableDiffusionExecutor.isEasyCacheSupported(this)
    }

    override fun close() {
        // T096: Proper cleanup - cancel any ongoing generation, destroy native context, reset state
        if (cancellationRequested.get()) {
            cancellationRequested.set(false)
        }
        // If tests have overridden the native bridge, the JNI library may not be loaded
        // so avoid calling nativeDestroy to prevent UnsatisfiedLinkError. See override
        // helpers in the companion object.
        if (!Companion.nativeBridgeOverriddenForTests && isNativeLibraryAvailable) {
            nativeDestroy(handle)
        }
        modelMetadata = null
        easyCacheSupported = null
    }

    private external fun nativeDestroy(handle: Long)

    private external fun nativeTxt2Img(
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

    private external fun nativeTxt2ImgArgb(
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

    private external fun nativeTxt2Vid(
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

    private external fun nativeSetProgressCallback(
            handle: Long,
            callback: VideoProgressCallback?,
    )

    private external fun nativeCancelGeneration(handle: Long)

    private external fun nativePrecomputeCondition(
            handle: Long,
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            clipSkip: Int,
    ): Array<Any?>?

    private external fun nativeTxt2VidWithPrecomputedCondition(
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

    private external fun nativeTxt2ImgWithPrecomputedCondition(
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

    private external fun nativeIsEasyCacheSupported(handle: Long): Boolean

    private fun bitmapToRgbBytes(bitmap: Bitmap): Triple<ByteArray, Int, Int> {
        return StableDiffusionOutputSupport.bitmapToRgbBytes(bitmap, rgbBytesThreadLocal)
    }

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
