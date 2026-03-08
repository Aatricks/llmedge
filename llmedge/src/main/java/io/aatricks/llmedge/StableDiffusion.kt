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

package io.aatricks.llmedge

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.os.Debug
import android.util.Log
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.InvalidGenerationParametersException
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.core.NativeBindingException
import io.aatricks.llmedge.core.NativeLibraryLoader
import io.aatricks.llmedge.core.UnsupportedModelException
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.huggingface.WanModelRegistry
import io.aatricks.llmedge.model.ModelFileValidator
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class StableDiffusion private constructor(private val handle: Long) : AutoCloseable {
    // Serialize concurrent generation calls - native library is not guaranteed to be reentrant.
    private val generationMutex = Mutex()
    private var modelMetadata: VideoModelMetadata? = null
    private val cancellationRequested = AtomicBoolean(false)
    private val rgbBytesThreadLocal = ThreadLocal<ByteArray>()
    // Reusable pixel buffer for txt2img RGB→ARGB conversion
    private var txt2imgPixelBuffer: IntArray? = null

    @Volatile private var cachedProgressCallback: VideoProgressCallback? = null

    @Volatile private var lastGenerationMetrics: GenerationMetrics? = null
    private val nativeBridge: NativeBridge = Companion.nativeBridgeProvider(this)

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
        private const val MIN_FRAME_BATCH = 4
        private const val MAX_FRAME_BATCH = 8

        private val isAndroidLogAvailable: Boolean =
                try {
                    Class.forName("android.util.Log")
                    true
                } catch (_: Throwable) {
                    false
                }

        // Cache reflected Log methods to avoid Class.forName + getMethod on every call
        private val cachedLogD: java.lang.reflect.Method? by lazy {
            try { Class.forName("android.util.Log").getMethod("d", String::class.java, String::class.java) } catch (_: Throwable) { null }
        }
        private val cachedLogI: java.lang.reflect.Method? by lazy {
            try { Class.forName("android.util.Log").getMethod("i", String::class.java, String::class.java) } catch (_: Throwable) { null }
        }
        private val cachedLogW: java.lang.reflect.Method? by lazy {
            try { Class.forName("android.util.Log").getMethod("w", String::class.java, String::class.java) } catch (_: Throwable) { null }
        }
        private val cachedLogE: java.lang.reflect.Method? by lazy {
            try { Class.forName("android.util.Log").getMethod("e", String::class.java, String::class.java, Throwable::class.java) } catch (_: Throwable) { null }
        }

        private fun logD(tag: String, message: String) {
            val method = cachedLogD
            if (method != null) {
                try { method.invoke(null, tag, message) } catch (_: Throwable) { println("D/$tag: $message") }
            } else {
                println("D/$tag: $message")
            }
        }

        private fun logI(tag: String, message: String) {
            val method = cachedLogI
            if (method != null) {
                try { method.invoke(null, tag, message) } catch (_: Throwable) { println("I/$tag: $message") }
            } else {
                println("I/$tag: $message")
            }
        }

        private fun logW(tag: String, message: String) {
            val method = cachedLogW
            if (method != null) {
                try { method.invoke(null, tag, message) } catch (_: Throwable) { println("W/$tag: $message") }
            } else {
                println("W/$tag: $message")
            }
        }

        private fun logE(tag: String, message: String, throwable: Throwable? = null) {
            val method = cachedLogE
            if (method != null) {
                try { method.invoke(null, tag, message, throwable) } catch (_: Throwable) {
                    System.err.println("E/$tag: $message")
                    throwable?.printStackTrace()
                }
            } else {
                System.err.println("E/$tag: $message")
                throwable?.printStackTrace()
            }
        }

        // Dummy instance used to invoke static native methods that are now at the class level.
        private val staticInvoker: StableDiffusion by lazy { StableDiffusion(0L) }

        @Volatile private var isNativeLibraryAvailable: Boolean
        // Flag set by tests when overriding the native bridge to a test mock so we avoid
        // calling actual JNI functions like nativeDestroy during Android instrumentation tests.
        private var nativeBridgeOverriddenForTests: Boolean = false

        // T098: Model metadata cache to avoid re-parsing GGUF headers
        private val metadataCache = mutableMapOf<String, VideoModelMetadata>()

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
                    // Array layout: [float[] cross, int[] crossDims, float[] vector, int[]
                    // vectorDims, float[] concat, int[] concatDims]
                    val cross = raw.getOrNull(0) as? FloatArray
                    val crossDims = raw.getOrNull(1) as? IntArray
                    val vector = raw.getOrNull(2) as? FloatArray
                    val vectorDims = raw.getOrNull(3) as? IntArray
                    val concat = raw.getOrNull(4) as? FloatArray
                    val concatDims = raw.getOrNull(5) as? IntArray
                    return PrecomputedCondition(
                            cCrossAttn = cross,
                            cCrossAttnDims = crossDims,
                            cVector = vector,
                            cVectorDims = vectorDims,
                            cConcat = concat,
                            cConcatDims = concatDims
                    )
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
                    val condArr =
                            cond?.let {
                                arrayOf<Any?>(
                                        it.cCrossAttn,
                                        it.cCrossAttnDims,
                                        it.cVector,
                                        it.cVectorDims,
                                        it.cConcat,
                                        it.cConcatDims
                                )
                            }
                    val uncondArr =
                            uncond?.let {
                                arrayOf<Any?>(
                                        it.cCrossAttn,
                                        it.cCrossAttnDims,
                                        it.cVector,
                                        it.cVectorDims,
                                        it.cConcat,
                                        it.cConcatDims
                                )
                            }
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
                            cond = condArr,
                            uncond = uncondArr,
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
                    val condArr =
                            cond?.let {
                                arrayOf<Any?>(
                                        it.cCrossAttn,
                                        it.cCrossAttnDims,
                                        it.cVector,
                                        it.cVectorDims,
                                        it.cConcat,
                                        it.cConcatDims
                                )
                            }
                    val uncondArr =
                            uncond?.let {
                                arrayOf<Any?>(
                                        it.cCrossAttn,
                                        it.cCrossAttnDims,
                                        it.cVector,
                                        it.cVectorDims,
                                        it.cConcat,
                                        it.cConcatDims
                                )
                            }
                    return instance.nativeTxt2ImgWithPrecomputedCondition(
                            handle,
                            prompt,
                            negative,
                            width,
                            height,
                            steps,
                            cfg,
                            seed,
                            condArr,
                            uncondArr,
                            easyCacheEnabled,
                            easyCacheReuseThreshold,
                            easyCacheStartPercent,
                            easyCacheEndPercent
                    )
                }
            }
        }

        @Volatile
        private var nativeBridgeProvider: (StableDiffusion) -> NativeBridge =
                defaultNativeBridgeProvider

        init {
            val disableNativeLoad = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")
            isNativeLibraryAvailable = !disableNativeLoad
            if (disableNativeLoad) {
                println("[StableDiffusion] Native load disabled via llmedge.disableNativeLoad=true")
            } else {
                NativeLibraryLoader.ensureStableDiffusionLoaded(
                    required = true,
                    onDebug = { message -> Log.d(LOG_TAG, message) },
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
            nativeBridgeProvider = provider
            nativeBridgeOverriddenForTests = true
        }

        internal fun resetNativeBridgeForTests() {
            nativeBridgeProvider = defaultNativeBridgeProvider
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
        private external fun nativeCreate(
                modelPath: String,
                vaePath: String?,
                t5xxlPath: String?,
                taesdPath: String?,
                nThreads: Int,
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
        private fun computeEffectiveSequentialLoad(
                context: Context,
                resolvedModelPath: String,
                sequentialLoad: Boolean?,
                preferPerformanceMode: Boolean,
                activityManagerOverride: ActivityManager? = null,
        ): Pair<Boolean, Long> {
            val memoryInfo = ActivityManager.MemoryInfo()
            val activityManager =
                    activityManagerOverride
                            ?: (context.getSystemService(Context.ACTIVITY_SERVICE) as
                                    ActivityManager)
            activityManager.getMemoryInfo(memoryInfo)
            val totalRamGB = memoryInfo.totalMem / (1024L * 1024L * 1024L)

            if (sequentialLoad != null) {
                return Pair(sequentialLoad, 0L)
            }

            var estimatedParamBytes = 0L
            try {
                val vulkanCount = nativeGetVulkanDeviceCount()
                val devIdx = if (vulkanCount > 0) 0 else -1
                estimatedParamBytes = estimateModelParamsMemoryBytes(resolvedModelPath, devIdx)
            } catch (t: Throwable) {
                estimatedParamBytes = 0L
            }

            val rt = Runtime.getRuntime()
            val heapUsed = rt.totalMemory() - rt.freeMemory()
            val heapMax = rt.maxMemory()
            val heapAvail = (heapMax - heapUsed).coerceAtLeast(0L)
            val sysAvail = memoryInfo.availMem

            val heapThresholdFactor = if (preferPerformanceMode) 0.9 else 0.75
            val sysThresholdFactor = if (preferPerformanceMode) 0.9 else 0.6

            val heapSeqNeeded =
                    (estimatedParamBytes > 0) &&
                            estimatedParamBytes.toDouble() >
                                    heapAvail.toDouble() * heapThresholdFactor
            val sysSeqNeeded =
                    (estimatedParamBytes > 0) &&
                            estimatedParamBytes.toDouble() >
                                    sysAvail.toDouble() * sysThresholdFactor
            val lowRamHint = (totalRamGB < 8)

            val effectiveSequentialLoad = lowRamHint || heapSeqNeeded || sysSeqNeeded
            return Pair(effectiveSequentialLoad, estimatedParamBytes)
        }

        private suspend fun inferVideoModelMetadata(
                resolvedModelPath: String,
                modelId: String?,
                explicitFilename: String?,
        ): VideoModelMetadata {
            android.util.Log.d(
                    LOG_TAG,
                    "inferVideoModelMetadata called: path=$resolvedModelPath, exists=${File(resolvedModelPath).exists()}"
            )

            // T098: Check cache first to avoid re-parsing GGUF
            val cacheKey = resolvedModelPath
            metadataCache[cacheKey]?.let {
                return it
            }

            val filename = explicitFilename ?: resolvedModelPath.substringAfterLast('/')
            val lowerName = filename.lowercase(Locale.US)
            val tags = mutableSetOf<String>()

            // Try to read from GGUF metadata first
            var architecture: String? = null
            var parameterCount: String? = null
            var modelType: String? = null

            // Skip GGUF parsing for video models - they use a different GGUF format
            // that llama.cpp's parser can't read. Rely on filename-based detection instead.
            android.util.Log.d(
                    LOG_TAG,
                    "Using filename-based video model detection for: $resolvedModelPath"
            )

            // Fallback to filename-based detection
            if (architecture == null) {
                architecture =
                        when {
                            !modelId.isNullOrBlank() -> modelId
                            lowerName.contains("hunyuan") -> "hunyuan_video"
                            lowerName.contains("wan") -> "wan"
                            else -> null
                        }
            }

            if (modelType == null) {
                modelType =
                        when {
                            lowerName.contains("ti2v") -> "ti2v"
                            lowerName.contains("i2v") -> "i2v"
                            lowerName.contains("t2v") -> "t2v"
                            else -> null
                        }
            }

            if (parameterCount == null) {
                parameterCount =
                        when {
                            lowerName.contains("1.3b") || lowerName.contains("1_3b") -> "1.3B"
                            lowerName.contains("5b") || lowerName.contains("5_b") -> "5B"
                            lowerName.contains("14b") || lowerName.contains("14_b") -> "14B"
                            else -> null
                        }
            }

            // Determine mobile support based on parameter count
            val mobileSupported =
                    when (parameterCount) {
                        "1.3B", "5B" -> true
                        "14B" -> false
                        else -> true // Unknown models assumed supported (will fail at load time if
                    // too large)
                    }

            // Build tags
            if (lowerName.contains("wan")) tags += "wan"
            if (lowerName.contains("video") || modelType in listOf("t2v", "i2v", "ti2v")) {
                tags += "text-to-video"
            }
            if (lowerName.contains("hunyuan")) tags += "hunyuan"

            val metadata =
                    VideoModelMetadata(
                            architecture = architecture,
                            modelType = modelType,
                            parameterCount = parameterCount,
                            mobileSupported = mobileSupported,
                            tags = tags,
                            filename = filename,
                    )

            // T098: Cache the metadata for future loads
            metadataCache[cacheKey] = metadata
            return metadata
        }

        private fun validateResolvedAssets(
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
                forceVulkan: Boolean = false,
                preferPerformanceMode: Boolean = false,
                token: String? = null,
                forceDownload: Boolean = false,
                flowShift: Float = Float.POSITIVE_INFINITY,
                loraModelDir: String? = null,
                loraApplyMode: LoraApplyMode = LoraApplyMode.AUTO,
        ): StableDiffusion =
                withContext(Dispatchers.IO) {
                    var resolvedModelPath: String
                    var resolvedVaePath: String?
                    var resolvedT5xxlPath: String?

                    if (modelPath != null) {
                        resolvedModelPath = modelPath
                        resolvedVaePath = vaePath
                        resolvedT5xxlPath = t5xxlPath
                    } else if (modelId != null) {
                        try {
                            val possibleWan =
                                    WanModelRegistry.findById(context, modelId)
                                            ?: WanModelRegistry.findByModelIdPrefix(
                                                    context,
                                                    modelId.removePrefix("wan/")
                                            )
                            if (possibleWan != null) {
                                return@withContext loadFromHuggingFace(
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
                                        preferPerformanceMode = preferPerformanceMode,
                                        token = token,
                                        forceDownload = forceDownload,
                                        preferSystemDownloader = true,
                                        flowShift = flowShift,
                                )
                            }
                        } catch (t: Throwable) {
                            // Continue with default downloader
                        }

                        try {
                            val res =
                                    HuggingFaceHub.ensureModelOnDisk(
                                            context = context,
                                            modelId = modelId,
                                            revision = "main",
                                            preferredQuantizations = emptyList(),
                                            filename = filename,
                                            token = token,
                                            forceDownload = forceDownload,
                                            preferSystemDownloader = true,
                                            onProgress = null,
                                    )
                            resolvedModelPath = res.file.absolutePath
                            resolvedVaePath = vaePath
                            resolvedT5xxlPath = t5xxlPath
                        } catch (iae: IllegalArgumentException) {
                            iae.printStackTrace()
                            val alt =
                                    HuggingFaceHub.ensureRepoFileOnDisk(
                                            context = context,
                                            modelId = modelId,
                                            revision = "main",
                                            filename = filename,
                                            allowedExtensions =
                                                    listOf(
                                                            ".gguf",
                                                            ".safetensors",
                                                            ".ckpt",
                                                            ".pt",
                                                            ".bin"
                                                    ),
                                            token = token,
                                            forceDownload = forceDownload,
                                            preferSystemDownloader = true,
                                            onProgress = null,
                                    )
                            resolvedModelPath = alt.file.absolutePath
                            resolvedVaePath = vaePath
                            resolvedT5xxlPath = t5xxlPath
                        }
                    } else {
                        throw InvalidGenerationParametersException(
                                "Provide either modelPath or modelId"
                        )
                    }

                    validateResolvedAssets(
                            modelPath = resolvedModelPath,
                            vaePath = resolvedVaePath,
                            t5xxlPath = resolvedT5xxlPath,
                            taesdPath = taesdPath,
                            loraModelDir = loraModelDir,
                    )

                    // T100: Memory pressure detection before loading
                    android.util.Log.d(
                            LOG_TAG,
                            "inferVideoModelMetadata: resolvedModelPath=$resolvedModelPath, exists=${File(resolvedModelPath).exists()}, modelId=$modelId, filename=$filename"
                    )

                    val metadata =
                            inferVideoModelMetadata(
                                    resolvedModelPath = resolvedModelPath,
                                    modelId = modelId,
                                    explicitFilename = filename,
                            )

                    // Check if we're trying to load a 5B model on a low-memory device
                    if (metadata.parameterCount == "5B") {
                        val memoryInfo = ActivityManager.MemoryInfo()
                        val activityManager =
                                context.getSystemService(Context.ACTIVITY_SERVICE) as
                                        ActivityManager
                        activityManager.getMemoryInfo(memoryInfo)
                        val totalRamGB = memoryInfo.totalMem / (1024L * 1024L * 1024L)

                        if (totalRamGB < 8) {
                            android.util.Log.w(
                                    LOG_TAG,
                                    "Loading 5B model on device with ${totalRamGB}GB RAM. " +
                                            "Consider using 1.3B variant for better performance. " +
                                            "Generation may be slow or fail with OOM."
                            )
                        }
                    }

                    // Auto-detect sequential load for low memory devices (< 8GB RAM)
                    val memoryInfo = ActivityManager.MemoryInfo()
                    val activityManager =
                            context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                    activityManager.getMemoryInfo(memoryInfo)
                    val totalRamGB = memoryInfo.totalMem / (1024L * 1024L * 1024L)

                    // Use a combined heuristic for sequential loading which considers
                    // 1) device total RAM, 2) Java available heap, and 3) native model param size
                    val (effectiveSequentialLoad, estimatedParamBytes) =
                            computeEffectiveSequentialLoad(
                                    context,
                                    resolvedModelPath,
                                    sequentialLoad,
                                    preferPerformanceMode,
                            )

                    var effectiveOffloadToCpu = offloadToCpu
                    var effectiveKeepClipOnCpu = keepClipOnCpu
                    var effectiveKeepVaeOnCpu = keepVaeOnCpu
                    var chosenDevice = -1
                    var estimatedParams: Long = 0
                    var freeBytes: Long = 0

                    if (effectiveSequentialLoad) {
                        if (sequentialLoad == null) {
                            android.util.Log.i(
                                    LOG_TAG,
                                    "Enabling sequential load for low memory optimization"
                            )
                            effectiveOffloadToCpu = true
                            effectiveKeepClipOnCpu = true
                            effectiveKeepVaeOnCpu = true
                        } else {
                            android.util.Log.i(
                                    LOG_TAG,
                                    "Sequential load explicitly requested; keeping existing offload settings"
                            )
                        }
                    }

                    // Debug log: show the inputs that influenced the combined sequential load
                    try {
                        val rt2 = Runtime.getRuntime()
                        val heapUsed2 = rt2.totalMemory() - rt2.freeMemory()
                        val heapMax2 = rt2.maxMemory()
                        val heapAvail2 = (heapMax2 - heapUsed2).coerceAtLeast(0L)
                        val sysAvail2 = memoryInfo.availMem
                        val lowRam2 = (totalRamGB < 8)
                        android.util.Log.i(
                                LOG_TAG,
                                "SequentialLoad heuristic: preferPerformanceMode=$preferPerformanceMode, explicitParam=${sequentialLoad}, lowRam=$lowRam2, estimatedParamBytes=$estimatedParamBytes, heapAvailMB=${String.format("%.2f", heapAvail2/1024.0/1024.0)}, sysAvailMB=${String.format("%.2f", sysAvail2/1024.0/1024.0)}, heapSeqNeeded=${(estimatedParamBytes>0 && estimatedParamBytes > heapAvail2 * (if (preferPerformanceMode) 0.9 else 0.75))}, sysSeqNeeded=${(estimatedParamBytes>0 && estimatedParamBytes > sysAvail2 * (if (preferPerformanceMode) 0.9 else 0.6))}, effectiveSequentialLoad=$effectiveSequentialLoad"
                        )
                    } catch (t: Throwable) {
                        // ignore logging errors
                    }

                    // Auto-detection heuristic: if device Vulkan VRAM is too small for this model,
                    // and the caller didn't explicitly ask for offloadToCpu, enable it
                    // automatically
                    try {
                        if (!effectiveOffloadToCpu) {
                            if (forceVulkan) {
                                android.util.Log.i(
                                        LOG_TAG,
                                        "forceVulkan=true requested; skipping Vulkan VRAM heuristics and preferring GPU path"
                                )
                            }
                            val vulkanDevices = nativeGetVulkanDeviceCount()
                            if (vulkanDevices > 0) {
                                var maxTotal: Long = 0
                                for (i in 0 until vulkanDevices) {
                                    val mem = nativeGetVulkanDeviceMemory(i)
                                    if (mem != null && mem.size >= 2) {
                                        val total = mem[1]
                                        if (total > maxTotal) {
                                            maxTotal = total
                                            chosenDevice = i
                                        }
                                    }
                                }
                                if (chosenDevice >= 0) {
                                    estimatedParams =
                                            estimateModelParamsMemoryBytes(
                                                    resolvedModelPath,
                                                    chosenDevice
                                            )
                                    if (estimatedParams > 0) {
                                        val mem = nativeGetVulkanDeviceMemory(chosenDevice)
                                        if (mem != null && mem.size >= 2) {
                                            freeBytes = mem[0]
                                            val THRESHOLD = 0.9
                                            if (!forceVulkan &&
                                                            estimatedParams.toDouble() >
                                                                    freeBytes.toDouble() * THRESHOLD
                                            ) {
                                                android.util.Log.i(
                                                        LOG_TAG,
                                                        "Vulkan VRAM insufficient for model; enabling offload_to_cpu (estimated: ${
                                                String.format(
                                                    "%.2f",
                                                    estimatedParams / 1024.0 / 1024.0
                                                )
                                            } MB, free: ${String.format("%.2f", freeBytes / 1024.0 / 1024.0)} MB)"
                                                )
                                                effectiveOffloadToCpu = true
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } catch (t: Throwable) {
                        // Best-effort heuristic; on any failure we'll not change the caller's
                        // preference
                        android.util.Log.w(
                                LOG_TAG,
                                "Failed to query Vulkan VRAM or estimate model memory: ${t.message}"
                        )
                    }

                    // Log final effective flags before creating native handle - this aids debugging
                    Log.i(
                            LOG_TAG,
                            "Initializing StableDiffusion (effective): modelPath=$resolvedModelPath, " +
                                    "nThreads=$nThreads, sequentialLoad=$effectiveSequentialLoad, " +
                                    "offloadToCpu=$effectiveOffloadToCpu, keepClipOnCpu=$effectiveKeepClipOnCpu, " +
                                    "keepVaeOnCpu=$effectiveKeepVaeOnCpu, flashAttn=$flashAttn"
                    )
                    if (chosenDevice >= 0) {
                        Log.i(
                                LOG_TAG,
                                "Vulkan chosenDevice=$chosenDevice, estimatedModelParamsMB=${String.format("%.2f", estimatedParams / 1024.0 / 1024.0)}, freeMB=${String.format("%.2f", freeBytes / 1024.0 / 1024.0)}"
                        )
                    }

                    var handle =
                            try {
                                nativeCreate(
                                        resolvedModelPath,
                                        resolvedVaePath,
                                        resolvedT5xxlPath,
                                        taesdPath,
                                        nThreads,
                                        effectiveOffloadToCpu,
                                        effectiveKeepClipOnCpu,
                                        effectiveKeepVaeOnCpu,
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
                                        cause = e
                                )
                            }
                    // If we requested preferred GPU path but nativeCreate failed, retry with CPU
                    // offload
                    if (handle == 0L && forceVulkan) {
                        android.util.Log.w(
                                LOG_TAG,
                                "nativeCreate failed with forceVulkan=true; retrying with offloadToCpu=true as a fallback"
                        )
                        effectiveOffloadToCpu = true
                        effectiveKeepClipOnCpu = true
                        effectiveKeepVaeOnCpu = true
                        handle =
                                try {
                                    nativeCreate(
                                            resolvedModelPath,
                                            resolvedVaePath,
                                            resolvedT5xxlPath,
                                            taesdPath,
                                            nThreads,
                                            effectiveOffloadToCpu,
                                            effectiveKeepClipOnCpu,
                                            effectiveKeepVaeOnCpu,
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
                                            cause = e
                                    )
                                }
                    } 
                    if (handle == 0L) {
                        val errorMsg = buildString {
                            append("Failed to initialize Stable Diffusion context.")
                            if (taesdPath != null) append(" Custom TAE/TAEHV: $taesdPath.")
                            if (resolvedVaePath != null) append(" Custom VAE: $resolvedVaePath.")
                            append(" This often happens due to incompatible VAE/TAE weights or insufficient memory. Check logcat for [SmolSD] errors.")
                        }
                        throw ModelLoadException(resolvedModelPath, errorMsg)
                    }
                    val instance = StableDiffusion(handle)
                    instance.modelMetadata = metadata

                    // T095: Mobile compatibility check - reject 14B models
                    if (instance.modelMetadata?.mobileSupported == false) {
                        instance.close()
                        val paramCount = instance.modelMetadata?.parameterCount ?: "14B"
                        throw UnsupportedModelException(
                                "$paramCount models are not supported on mobile devices. " +
                                        "Please use 1.3B or 5B model variants instead. " +
                                        "14B models require 20-40GB RAM and are designed for desktop/server use only."
                        )
                    }

                    instance
                }

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
                withContext(Dispatchers.IO) {
                    val (modelRes, vaeRes, t5Res) =
                            HuggingFaceHub.ensureWanAssetsOnDisk(
                                    context = context,
                                    wanModelId = modelId,
                                    preferSystemDownloader = preferSystemDownloader,
                                    token = token,
                                    forceDownload = forceDownload,
                                    onProgress = { downloaded, total ->
                                        onProgress?.invoke(modelId, downloaded, total)
                                    },
                            )

                    validateResolvedAssets(
                            modelPath = modelRes.file.absolutePath,
                            vaePath = vaeRes?.file?.absolutePath,
                            t5xxlPath = t5Res?.file?.absolutePath,
                            taesdPath = taesdPath,
                            loraModelDir = loraModelDir,
                    )

                    val memoryInfo = ActivityManager.MemoryInfo()
                    val activityManager =
                            context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                    activityManager.getMemoryInfo(memoryInfo)
                    val totalRamGB = memoryInfo.totalMem / (1024L * 1024L * 1024L)

                    val (effectiveSequentialLoad, estimatedParamBytes) =
                            computeEffectiveSequentialLoad(
                                    context,
                                    modelRes.file.absolutePath,
                                    sequentialLoad,
                                    preferPerformanceMode
                            )

                    var effectiveOffloadToCpu = offloadToCpu
                    var effectiveKeepClipOnCpu = keepClipOnCpu
                    var effectiveKeepVaeOnCpu = keepVaeOnCpu

                    if (effectiveSequentialLoad) {
                        if (sequentialLoad == null) {
                            effectiveOffloadToCpu = true
                            effectiveKeepClipOnCpu = true
                            effectiveKeepVaeOnCpu = true
                        } else {
                            android.util.Log.i(
                                    LOG_TAG,
                                    "Sequential load explicitly requested for HF model; keeping existing offload settings"
                            )
                        }
                    }

                    try {
                        val rt2 = Runtime.getRuntime()
                        val heapUsed2 = rt2.totalMemory() - rt2.freeMemory()
                        val heapMax2 = rt2.maxMemory()
                        val heapAvail2 = (heapMax2 - heapUsed2).coerceAtLeast(0L)
                        val sysAvail2 = memoryInfo.availMem
                        val lowRam2 = (totalRamGB < 8)
                        android.util.Log.i(
                                LOG_TAG,
                                "SequentialLoad HF heuristic: preferPerformanceMode=$preferPerformanceMode, explicitParam=${sequentialLoad}, lowRam=$lowRam2, estimatedParamBytes=$estimatedParamBytes, heapAvailMB=${String.format("%.2f", heapAvail2/1024.0/1024.0)}, sysAvailMB=${String.format("%.2f", sysAvail2/1024.0/1024.0)}, heapSeqNeeded=${(estimatedParamBytes>0 && estimatedParamBytes > heapAvail2 * (if (preferPerformanceMode) 0.9 else 0.75))}, sysSeqNeeded=${(estimatedParamBytes>0 && estimatedParamBytes > sysAvail2 * (if (preferPerformanceMode) 0.9 else 0.6))}, effectiveSequentialLoad=$effectiveSequentialLoad"
                        )
                    } catch (t: Throwable) {
                        // ignore logging errors
                    }

                    var chosenDevice = -1
                    var estimatedParams: Long = -1
                    var freeBytes: Long = -1
                    if (!effectiveOffloadToCpu && forceVulkan) {
                        android.util.Log.i(
                                LOG_TAG,
                                "forceVulkan=true requested; skipping Vulkan VRAM heuristics and preferring GPU path for HF loaded model"
                        )
                    }
                    try {
                        if (!effectiveOffloadToCpu) {
                            val vulkanDevices = nativeGetVulkanDeviceCount()
                            if (vulkanDevices > 0) {
                                var maxTotal: Long = 0
                                for (i in 0 until vulkanDevices) {
                                    val mem = nativeGetVulkanDeviceMemory(i)
                                    if (mem != null && mem.size >= 2) {
                                        val total = mem[1]
                                        if (total > maxTotal) {
                                            maxTotal = total
                                            chosenDevice = i
                                        }
                                    }
                                }
                                if (chosenDevice >= 0) {
                                    estimatedParams =
                                            estimateModelParamsMemoryBytes(
                                                    modelRes.file.absolutePath,
                                                    chosenDevice
                                            )
                                    if (estimatedParams > 0) {
                                        val mem = nativeGetVulkanDeviceMemory(chosenDevice)
                                        if (mem != null && mem.size >= 2) {
                                            freeBytes = mem[0]
                                            val THRESHOLD = 0.9
                                            if (!forceVulkan &&
                                                            estimatedParams.toDouble() >
                                                                    freeBytes.toDouble() * THRESHOLD
                                            ) {
                                                android.util.Log.i(
                                                        LOG_TAG,
                                                        "Vulkan VRAM insufficient for HF model; enabling offload_to_cpu (estimated: ${String.format("%.2f", estimatedParams / 1024.0 / 1024.0)} MB, free: ${String.format("%.2f", freeBytes / 1024.0 / 1024.0)} MB)"
                                                )
                                                effectiveOffloadToCpu = true
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } catch (t: Throwable) {
                        android.util.Log.w(
                                LOG_TAG,
                                "Failed to query Vulkan VRAM or estimate HF model memory: ${t.message}"
                        )
                    }

                    // Debug log for model initialization choices
                    Log.i(
                            LOG_TAG,
                            "Initializing StableDiffusion from HF (effective): model=${modelRes.file.absolutePath}, nThreads=$nThreads, sequentialLoad=$effectiveSequentialLoad, offloadToCpu=$effectiveOffloadToCpu, keepClipOnCpu=$effectiveKeepClipOnCpu, keepVaeOnCpu=$effectiveKeepVaeOnCpu, flashAttn=$flashAttn"
                    )

                    var handle =
                            try {
                                nativeCreate(
                                        modelRes.file.absolutePath,
                                        vaeRes?.file?.absolutePath,
                                        t5Res?.file?.absolutePath,
                                        taesdPath,
                                        nThreads,
                                        effectiveOffloadToCpu,
                                        effectiveKeepClipOnCpu,
                                        effectiveKeepVaeOnCpu,
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
                                        cause = e
                                )
                            }
                    if (handle == 0L && forceVulkan) {
                        android.util.Log.w(
                                LOG_TAG,
                                "nativeCreate failed with forceVulkan=true; retrying with offloadToCpu=true as a fallback (HF)"
                        )
                        effectiveOffloadToCpu = true
                        effectiveKeepClipOnCpu = true
                        effectiveKeepVaeOnCpu = true
                        handle =
                                try {
                                    nativeCreate(
                                            modelRes.file.absolutePath,
                                            vaeRes?.file?.absolutePath,
                                            t5Res?.file?.absolutePath,
                                            taesdPath,
                                            nThreads,
                                            effectiveOffloadToCpu,
                                            effectiveKeepClipOnCpu,
                                            effectiveKeepVaeOnCpu,
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
                                            cause = e
                                    )
                                }
                    }
                    if (handle == 0L)
                            throw ModelLoadException(
                                    modelRes.file.absolutePath,
                                    "Failed to initialize Stable Diffusion context"
                            )
                    val instance = StableDiffusion(handle)
                    instance.modelMetadata =
                            inferVideoModelMetadata(
                                    resolvedModelPath = modelRes.file.absolutePath,
                                    modelId = modelId,
                                    explicitFilename = filename,
                            )
                    instance
                }
    }

    data class GenerateParams(
            val prompt: String,
            val negative: String = "",
            val width: Int = 512,
            val height: Int = 512,
            val steps: Int = 20,
            val cfgScale: Float = 7.0f,
            val seed: Long = 42L,
            val vaeTiling: Boolean = true,
            val easyCacheParams: EasyCacheParams = EasyCacheParams()
    )

    data class EasyCacheParams(
            val enabled: Boolean = false,
            val reuseThreshold: Float = 0.2f,
            val startPercent: Float = 0.15f,
            val endPercent: Float = 0.95f,
    )

    /** Sample methods for diffusion models. Maps to native sample_method_t enum values. */
    enum class SampleMethod(val id: Int) {
        /** Let native code choose the default for the model type */
        DEFAULT(0),
        /** Euler sampler - Default and recommended for DiT models (Flux/SD3/Wan) */
        EULER(1),
        /** Heun sampler - Higher quality, 2x computation. Works with all models. */
        HEUN(2),
        /** DPM2 sampler - Best for U-Net models (SD1.x/SD2.x/SDXL). Not recommended for Wan. */
        DPM2(3),
        /** DPM++ 2S Ancestral - Best for U-Net models. Not recommended for Wan. */
        DPMPP2S_A(4),
        /** DPM++ 2M - Best for U-Net models. Not recommended for Wan video generation. */
        DPMPP2M(5),
        /** DPM++ 2M v2 - Best for U-Net models. Not recommended for Wan. */
        DPMPP2MV2(6),
        /** IPNDM - Fast sampler */
        IPNDM(7),
        /** IPNDM v */
        IPNDM_V(8),
        /** Latent Consistency Models - Requires LCM-distilled models. NOT compatible with Wan. */
        LCM(9),
        /** DDIM Trailing */
        DDIM_TRAILING(10),
        /** TCD */
        TCD(11),
        /**
         * Euler Ancestral - Default for U-Net models (SD1.x/SD2.x/SDXL). May work with Wan but
         * EULER is preferred.
         */
        EULER_A(12);

        companion object {
            fun fromId(id: Int): SampleMethod = values().firstOrNull { it.id == id } ?: DEFAULT

            /** Samplers recommended for Wan video generation */
            val WAN_RECOMMENDED = listOf(DEFAULT, EULER, HEUN)

            /** Samplers that are NOT compatible with Wan (produce blank/noise output) */
            val WAN_INCOMPATIBLE = listOf(LCM, DPMPP2M, DPMPP2MV2, DPM2, DPMPP2S_A)
        }
    }

    /** Noise schedulers for diffusion models. Maps to native scheduler_t enum values. */
    enum class Scheduler(val id: Int) {
        /** Let native code choose the default scheduler */
        DEFAULT(0),
        /** Discrete scheduler */
        DISCRETE(1),
        /** Karras scheduler - Often better quality */
        KARRAS(2),
        /** Exponential scheduler */
        EXPONENTIAL(3),
        /** AYS scheduler */
        AYS(4),
        /** GITS scheduler */
        GITS(5),
        /** SGM Uniform scheduler */
        SGM_UNIFORM(6),
        /** Simple scheduler */
        SIMPLE(7),
        /** Smoothstep scheduler */
        SMOOTHSTEP(8);

        companion object {
            fun fromId(id: Int): Scheduler = values().firstOrNull { it.id == id } ?: DEFAULT

            /** Schedulers known to work reliably with Wan video generation */
            val WAN_RECOMMENDED = listOf(DEFAULT)
        }
    }

    // Legacy alias for backward compatibility
    @Deprecated("Use SampleMethod enum instead", ReplaceWith("SampleMethod"))
    val EULER_A = SampleMethod.EULER_A
    @Deprecated("Use SampleMethod enum instead", ReplaceWith("SampleMethod"))
    val DDIM = SampleMethod.DDIM_TRAILING
    @Deprecated("Use SampleMethod enum instead", ReplaceWith("SampleMethod"))
    val LCM = SampleMethod.LCM

    data class VideoGenerateParams(
            val prompt: String,
            val negative: String = "",
            val width: Int = 512,
            val height: Int = 512,
            val videoFrames: Int = 16,
            val steps: Int = 20,
            val cfgScale: Float = 7.0f,
            val seed: Long = -1L,
            val initImage: Bitmap? = null,
            val strength: Float = 0.8f,
            val vaceStrength: Float = 1.0f,
            val sampleMethod: SampleMethod = SampleMethod.DEFAULT,
            val scheduler: Scheduler = Scheduler.DEFAULT,
            val easyCacheParams: EasyCacheParams = EasyCacheParams()
    ) {
        /**
         * Calculate the actual number of frames that will be generated. Wan model uses formula:
         * actual_frames = (videoFrames-1)/4*4+1 Examples: 5→5, 8→5, 9→9, 10→9, 12→9, 13→13
         */
        fun actualFrameCount(): Int = (videoFrames - 1) / 4 * 4 + 1

        fun validate(): Result<Unit> = runCatching {
            require(prompt.isNotBlank()) { "Prompt cannot be blank" }
            require(width % 64 == 0 && width in 128..960) {
                "Width must be a multiple of 64 in range 128..960"
            }
            require(height % 64 == 0 && height in 128..960) {
                "Height must be a multiple of 64 in range 128..960"
            }
            // Wan model uses formula: actual_frames = (videoFrames-1)/4*4+1
            // So 1-4 -> 1 frame, 5-8 -> 5 frames, 9-12 -> 9 frames, etc.
            require(videoFrames in 5..64) {
                "Frame count must be between 5 and 64. Note: Wan model rounds to (n-1)/4*4+1, so use 5+ for multiple frames"
            }
            require(steps in 1..50) { "Steps must be between 1 and 50" }
            require(cfgScale in 1.0f..15.0f) { "CFG scale must be between 1.0 and 15.0" }
            require(strength in 0.0f..1.0f) { "Strength must be between 0.0 and 1.0" }
            require(seed >= -1L) { "Seed must be -1 or non-negative" }

            // Validate init image + strength consistency (I2V mode)
            if (initImage != null) {
                require(strength > 0.0f) {
                    "When initImage is provided (I2V mode), strength must be > 0.0"
                }
            }
        }

        fun withPrompt(prompt: String): VideoGenerateParams = copy(prompt = prompt)

        companion object {
            fun default(prompt: String = "") = VideoGenerateParams(prompt = prompt)

            /**
             * Get the recommended videoFrames value to generate exactly N frames. Since Wan uses
             * (n-1)/4*4+1, to get exactly N frames you need: 1 frame → 1-4, 5 frames → 5-8, 9
             * frames → 9-12, etc.
             */
            fun recommendedFrameInput(desiredFrames: Int): Int {
                require(desiredFrames >= 1) { "Desired frames must be at least 1" }
                // Reverse the formula: to get N, input N is fine if N = (N-1)/4*4+1
                // Otherwise input N+3 at most
                return if (desiredFrames == 1) 1 else ((desiredFrames - 1) / 4) * 4 + 5
            }
        }
    }

    enum class LoraApplyMode(val id: Int) {
        AUTO(0),
        IMMEDIATELY(1),
        AT_RUNTIME(2);
        companion object {
            fun fromId(id: Int): LoraApplyMode = values().firstOrNull { it.id == id } ?: AUTO
        }
    }

    data class GenerationMetrics(
            val totalTimeSeconds: Float,
            val framesPerSecond: Float,
            val timePerStep: Float,
            val peakMemoryUsageMb: Long,
            val vulkanEnabled: Boolean,
            val frameConversionTimeSeconds: Float = 0f,
    ) {
        val averageFrameTime: Float
            get() = if (framesPerSecond > 0f) 1f / framesPerSecond else 0f

        val stepsPerSecond: Float
            get() = if (timePerStep > 0f) 1f / timePerStep else 0f

        val throughput: String
            get() = String.format("%.2f fps", framesPerSecond)

        fun toPrettyString(): String =
                """
            Total time: ${String.format("%.2f", totalTimeSeconds)}s
            Throughput: ${String.format("%.2f", framesPerSecond)} fps
            Average time/step: ${String.format("%.3f", timePerStep)}s
            Peak memory: ${peakMemoryUsageMb}MB
            Vulkan: ${if (vulkanEnabled) "enabled" else "disabled"}
            Frame conversion: ${String.format("%.2f", frameConversionTimeSeconds)}s
        """.trimIndent()
    }

    fun interface VideoProgressCallback {
        fun onProgress(
                step: Int,
                totalSteps: Int,
                currentFrame: Int,
                totalFrames: Int,
                timePerStep: Float,
        )
    }

    /**
     * Container for precomputed text-conditioning arrays returned by the native
     * `sd_precompute_condition` API. Each field is optional and will be null if that tensor is not
     * used for the given model / prompt.
     */
    data class PrecomputedCondition(
            val cCrossAttn: FloatArray? = null,
            val cCrossAttnDims: IntArray? = null,
            val cVector: FloatArray? = null,
            val cVectorDims: IntArray? = null,
            val cConcat: FloatArray? = null,
            val cConcatDims: IntArray? = null,
    )

    internal data class VideoModelMetadata(
            val architecture: String?,
            val modelType: String?,
            val parameterCount: String?,
            val mobileSupported: Boolean,
            val tags: Set<String>,
            val filename: String,
    )

    internal fun updateModelMetadata(metadata: VideoModelMetadata?) {
        modelMetadata = metadata
    }

    fun isVideoModel(): Boolean {
        val metadata = modelMetadata ?: return false
        return VideoModelDetector.isVideoModel(metadata)
    }

    suspend fun txt2vid(
            params: VideoGenerateParams,
            onProgress: VideoProgressCallback? = null,
    ): List<Bitmap> =
            withContext(Dispatchers.IO) {
                check(isNativeLibraryAvailable) {
                    "Video generation is unavailable on this platform"
                }
                params.validate().getOrThrow()
                check(isVideoModel()) { "Loaded model is not a video model (use txt2img instead)" }

                // T101: Context size capping based on model size
                val maxFrames =
                        when (modelMetadata?.parameterCount) {
                            "5B" -> 32 // 5B models limited to 32 frames on mobile
                            else -> 64 // 1.3B and unknown models can use full 64 frames
                        }
                require(params.videoFrames <= maxFrames) {
                    "Model ${modelMetadata?.parameterCount ?: "unknown"} supports maximum $maxFrames frames. " +
                            "Requested ${params.videoFrames} frames. Use a smaller model or reduce frame count."
                }

                val estimatedBytes =
                        estimateFrameFootprintBytes(
                                width = params.width,
                                height = params.height,
                                frameCount = params.videoFrames,
                        )
                warnIfLowMemory(estimatedBytes)

                val (initBytes, initWidth, initHeight) =
                        params.initImage?.let { bitmapToRgbBytes(it) } ?: Triple(null, 0, 0)

                val tempCallback = onProgress
                if (tempCallback != null) {
                    nativeBridge.setProgressCallback(handle, tempCallback)
                }

                val startNanos = System.nanoTime()
                val memoryBefore = readNativeMemoryMb()
                var frameBytes =
                        try {
                            generationMutex.withLock {
                                cancellationRequested.set(false)
                                nativeBridge.txt2vid(
                                        handle,
                                        params.prompt,
                                        params.negative,
                                        params.width,
                                        params.height,
                                        params.videoFrames,
                                        params.steps,
                                        params.cfgScale,
                                        params.seed,
                                        params.sampleMethod,
                                        params.scheduler,
                                        params.strength,
                                        initBytes,
                                        initWidth,
                                        initHeight,
                                        params.vaceStrength,
                                        params.easyCacheParams.enabled,
                                        params.easyCacheParams.reuseThreshold,
                                        params.easyCacheParams.startPercent,
                                        params.easyCacheParams.endPercent,
                                )
                                        ?: throw InferenceFailedException(
                                                operation = "Stable Diffusion video generation",
                                                detail = "The native runtime reported a generation failure."
                                        )
                            }
                        } catch (t: Throwable) {
                            if (cancellationRequested.get()) {
                                cancellationRequested.set(false)
                                throw CancellationException("Video generation cancelled", t)
                            }
                            throw t
                        } finally {
                            cancellationRequested.set(false)
                        }

                if (frameBytes.isEmpty()) {
                    throw InferenceFailedException(
                            operation = "Stable Diffusion video generation",
                            detail = "The native runtime returned no frames."
                    )
                }

                // Note: Wan model calculates actual frames as (n-1)/4*4+1
                val expectedFrames = params.actualFrameCount()
                if (frameBytes.size != expectedFrames) {
                    Log.w(
                            LOG_TAG,
                            "Expected $expectedFrames frames (formula: (${params.videoFrames}-1)/4*4+1) but received ${frameBytes.size}",
                    )
                }

                // Heuristic: if native output appears to be fully black/near zero, attempt
                // a channel-swap fallback (common when channel ordering is reversed or
                // when a plugin returns bytes in a different layout). This helps surface
                // real frames instead of blank images in environments with inconsistent
                // native outputs.
                fun computeAvgBrightness(bytes: ByteArray): Double {
                    var s = 0L
                    var i = 0
                    val totalPixels = (bytes.size / 3).coerceAtLeast(1)
                    while (i + 2 < bytes.size) {
                        val r = bytes[i++].toInt() and 0xFF
                        val g = bytes[i++].toInt() and 0xFF
                        val b = bytes[i++].toInt() and 0xFF
                        s += (r + g + b) / 3
                    }
                    return s.toDouble() / totalPixels
                }

                val avg = frameBytes.map { computeAvgBrightness(it) }.average()
                Log.d(
                        LOG_TAG,
                        "Video frame analysis: ${frameBytes.size} frames, avg brightness=$avg, first frame size=${frameBytes.firstOrNull()?.size ?: 0}"
                )

                if (avg < 1.0) {
                    Log.w(
                            LOG_TAG,
                            "Detected potentially black frames (avg brightness < 1.0), attempting channel swap..."
                    )
                    // Try swapping R and B channels
                    val swapped =
                            frameBytes
                                    .map { bytes ->
                                        val out = ByteArray(bytes.size)
                                        var j = 0
                                        var k = 0
                                        while (k + 2 < bytes.size) {
                                            val r = bytes[k]
                                            val g = bytes[k + 1]
                                            val b = bytes[k + 2]
                                            out[j++] = b
                                            out[j++] = g
                                            out[j++] = r
                                            k += 3
                                        }
                                        out
                                    }
                                    .toTypedArray()
                    val swappedAvg = swapped.map { computeAvgBrightness(it) }.average()
                    Log.d(LOG_TAG, "After BGR swap: avg brightness=$swappedAvg")
                    if (swappedAvg > avg) {
                        frameBytes = swapped
                        Log.w(
                                LOG_TAG,
                                "Swapped RGB->BGR for video frames to recover non-black output"
                        )
                    } else if (avg < 0.1) {
                        // Still very dark - log raw byte samples for debugging
                        val sample = frameBytes.firstOrNull()
                        if (sample != null && sample.size >= 30) {
                            val sampleBytes = sample.take(30).map { it.toInt() and 0xFF }
                            Log.e(
                                    LOG_TAG,
                                    "Frame appears completely black. First 30 bytes: $sampleBytes"
                            )
                        }
                    }
                }

                val conversionStart = System.nanoTime()
                val bitmaps = convertFramesToBitmaps(frameBytes, params.width, params.height)
                val conversionSeconds = ((System.nanoTime() - conversionStart) / 1_000_000_000f)
                val totalSeconds = ((System.nanoTime() - startNanos) / 1_000_000_000f)
                val memoryAfter = readNativeMemoryMb()

                lastGenerationMetrics =
                        GenerationMetrics(
                                totalTimeSeconds = totalSeconds,
                                framesPerSecond =
                                        if (totalSeconds > 0f) bitmaps.size / totalSeconds else 0f,
                                timePerStep =
                                        if (params.steps > 0) totalSeconds / params.steps else 0f,
                                peakMemoryUsageMb = maxOf(memoryBefore, memoryAfter),
                                vulkanEnabled = false,
                                frameConversionTimeSeconds = conversionSeconds,
                        )

                warnIfLowMemory(estimatedBytes)

                if (tempCallback != null) {
                    nativeBridge.setProgressCallback(handle, cachedProgressCallback)
                }

                bitmaps
            }

    fun setProgressCallback(callback: VideoProgressCallback?) {
        cachedProgressCallback = callback
        if (!isNativeLibraryAvailable) return
        nativeBridge.setProgressCallback(handle, callback)
    }

    fun cancelGeneration() {
        cancellationRequested.set(true)
        if (!isNativeLibraryAvailable) return
        nativeBridge.cancelGeneration(handle)
    }

    fun getLastGenerationMetrics(): GenerationMetrics? = lastGenerationMetrics

    suspend fun txt2img(params: GenerateParams): Bitmap =
            // Use Dispatchers.Default for CPU-bound generation to prefer a CPU-optimized
            // thread pool and reduce context-switching/stack allocations compared to IO.
            withContext(Dispatchers.Default) {
                val bytes =
                        generationMutex.withLock {
                            nativeBridge.txt2img(
                                    handle,
                                    params.prompt,
                                    params.negative,
                                    params.width,
                                    params.height,
                                    params.steps,
                                    params.cfgScale,
                                    params.seed,
                                    params.vaeTiling,
                                    params.easyCacheParams.enabled,
                                    params.easyCacheParams.reuseThreshold,
                                    params.easyCacheParams.startPercent,
                                    params.easyCacheParams.endPercent
                            )
                                    ?: throw InferenceFailedException(
                                            operation = "Stable Diffusion image generation",
                                            detail = "The native runtime reported a generation failure."
                                    )
                        }

                // Convert raw RGB bytes to Bitmap
                val bmp = Bitmap.createBitmap(params.width, params.height, Bitmap.Config.ARGB_8888)
                // Convert RGB to ARGB using reusable pixel buffer
                val rgb = bytes
                val expectedMin = params.width * params.height * 3
                if (rgb.size < expectedMin) {
                    Log.w(
                            LOG_TAG,
                            "txt2img returned short RGB buffer: size=${rgb.size}, expectedAtLeast=$expectedMin (w=${params.width}, h=${params.height})"
                    )
                }
                val pixelCount = params.width * params.height
                val pixels = txt2imgPixelBuffer.let { buf ->
                    if (buf != null && buf.size >= pixelCount) buf
                    else IntArray(pixelCount).also { txt2imgPixelBuffer = it }
                }
                var idx = 0
                var p = 0
                while (idx + 2 < rgb.size && p < pixelCount) {
                    pixels[p] = (0xFF shl 24) or
                            ((rgb[idx].toInt() and 0xFF) shl 16) or
                            ((rgb[idx + 1].toInt() and 0xFF) shl 8) or
                            (rgb[idx + 2].toInt() and 0xFF)
                    idx += 3
                    p += 1
                }
                bmp.setPixels(pixels, 0, params.width, 0, 0, params.width, params.height)
                bmp
            }

    fun isEasyCacheSupported(): Boolean {
        if (!isNativeLibraryAvailable) return false
        return nativeIsEasyCacheSupported(handle)
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
        val safeBitmap =
                if (bitmap.config == Bitmap.Config.ARGB_8888) {
                    bitmap
                } else {
                    bitmap.copy(Bitmap.Config.ARGB_8888, false)
                }
        val width = safeBitmap.width
        val height = safeBitmap.height
        val pixels = IntArray(width * height)
        safeBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val requiredSize = width * height * 3
        var rgb = rgbBytesThreadLocal.get()
        if (rgb == null || rgb.size < requiredSize) {
            rgb = ByteArray(requiredSize)
            rgbBytesThreadLocal.set(rgb)
        }
        var rgbIndex = 0
        for (pixel in pixels) {
            rgb[rgbIndex++] = ((pixel shr 16) and 0xFF).toByte()
            rgb[rgbIndex++] = ((pixel shr 8) and 0xFF).toByte()
            rgb[rgbIndex++] = (pixel and 0xFF).toByte()
        }
        return Triple(rgb, width, height)
    }

    private fun convertFramesToBitmaps(
            frameBytes: Array<ByteArray>,
            width: Int,
            height: Int,
    ): List<Bitmap> {
        val batchSize = determineBatchSize(frameBytes.size)
        val bitmaps = ArrayList<Bitmap>(frameBytes.size)
        // Reuse a single pixel buffer across all frames in this batch
        val pixelBuffer = IntArray(width * height)
        var index = 0
        while (index < frameBytes.size) {
            val end = min(index + batchSize, frameBytes.size)
            for (i in index until end) {
                // Make a defensive copy of the incoming bytes to protect against
                // native bridges that reuse or alias the same ByteArray for multiple
                // frames. Cloning ensures we snapshot the bytes for the frame and
                // prevents later mutations from affecting previously created
                // Bitmaps.
                val bytesCopy = frameBytes[i].clone()
                bitmaps += rgbBytesToBitmap(bytesCopy, width, height, pixelBuffer)
            }
            val remaining = frameBytes.size - end
            if (remaining > 0) {
                warnIfLowMemory(estimateFrameFootprintBytes(width, height, remaining))
            }
            index = end
        }
        return bitmaps
    }

    /**
     * Wrapper that calls the native PrecomputeCondition API and converts to a Kotlin type.
     *
     * This computes a single conditioning for the provided [prompt]. If you intend to use CFG
     * ($\text{cfgScale} \neq 1$), also precompute an unconditional/negative conditioning (e.g. with
     * an empty prompt or your negative prompt) and pass it as `uncond` to
     * [txt2VidWithPrecomputedCondition].
     */
    fun precomputeCondition(
            prompt: String,
            negative: String = "",
            width: Int = 512,
            height: Int = 512,
            clipSkip: Int = -1
    ): PrecomputedCondition? {
        // Delegate to the native bridge (which will call into JNI nativePrecomputeCondition by
        // default)
        return nativeBridge.precomputeCondition(handle, prompt, negative, width, height, clipSkip)
    }

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
    ): List<Bitmap> =
            withContext(Dispatchers.IO) {
                check(isNativeLibraryAvailable) {
                    "Video generation is unavailable on this platform"
                }
                params.validate().getOrThrow()
                check(isVideoModel()) { "Loaded model is not a video model (use txt2img instead)" }

                val estimatedBytes =
                        estimateFrameFootprintBytes(
                                width = params.width,
                                height = params.height,
                                frameCount = params.videoFrames,
                        )
                warnIfLowMemory(estimatedBytes)

                val (initBytes, initWidth, initHeight) =
                        params.initImage?.let { bitmapToRgbBytes(it) } ?: Triple(null, 0, 0)

                val tempCallback = onProgress
                if (tempCallback != null) {
                    nativeBridge.setProgressCallback(handle, tempCallback)
                }

                val startNanos = System.nanoTime()
                val memoryBefore = readNativeMemoryMb()
                val frameBytes =
                        try {
                            generationMutex.withLock {
                                cancellationRequested.set(false)
                                nativeBridge.txt2vidWithPrecomputedCondition(
                                        handle,
                                        params.prompt,
                                        params.negative,
                                        params.width,
                                        params.height,
                                        params.videoFrames,
                                        params.steps,
                                        params.cfgScale,
                                        params.seed,
                                        params.sampleMethod,
                                        params.scheduler,
                                        params.strength,
                                        initBytes,
                                        initWidth,
                                        initHeight,
                                        cond,
                                        uncond,
                                        params.vaceStrength,
                                        params.easyCacheParams.enabled,
                                        params.easyCacheParams.reuseThreshold,
                                        params.easyCacheParams.startPercent,
                                        params.easyCacheParams.endPercent,
                                )
                                        ?: throw InferenceFailedException(
                                                operation = "Stable Diffusion video generation",
                                                detail = "The native runtime reported a generation failure."
                                        )
                            }
                        } catch (t: Throwable) {
                            if (cancellationRequested.get()) {
                                cancellationRequested.set(false)
                                throw CancellationException("Video generation cancelled", t)
                            }
                            throw t
                        } finally {
                            cancellationRequested.set(false)
                        }

                if (frameBytes.isEmpty()) {
                    throw InferenceFailedException(
                            operation = "Stable Diffusion video generation",
                            detail = "The native runtime returned no frames."
                    )
                }

                // Note: Wan model calculates actual frames as (n-1)/4*4+1
                val expectedFrames = params.actualFrameCount()
                if (frameBytes.size != expectedFrames) {
                    Log.w(
                            LOG_TAG,
                            "Expected $expectedFrames frames (formula: (${params.videoFrames}-1)/4*4+1) but received ${frameBytes.size}",
                    )
                }

                val conversionStart = System.nanoTime()
                val bitmaps = convertFramesToBitmaps(frameBytes, params.width, params.height)
                val conversionSeconds = ((System.nanoTime() - conversionStart) / 1_000_000_000f)
                val totalSeconds = ((System.nanoTime() - startNanos) / 1_000_000_000f)
                val memoryAfter = readNativeMemoryMb()

                lastGenerationMetrics =
                        GenerationMetrics(
                                totalTimeSeconds = totalSeconds,
                                framesPerSecond =
                                        if (totalSeconds > 0f) bitmaps.size / totalSeconds else 0f,
                                timePerStep =
                                        if (params.steps > 0) totalSeconds / params.steps else 0f,
                                peakMemoryUsageMb = maxOf(memoryBefore, memoryAfter),
                                vulkanEnabled = false,
                                frameConversionTimeSeconds = conversionSeconds,
                        )

                warnIfLowMemory(estimatedBytes)

                if (tempCallback != null) {
                    nativeBridge.setProgressCallback(handle, cachedProgressCallback)
                }

                bitmaps
            }

    // Compat wrapper for callers expecting the 3-arg signature.
    private fun rgbBytesToBitmap(bytes: ByteArray, width: Int, height: Int): Bitmap {
        return io.aatricks.llmedge.vision.ImageUtils.rgbBytesToBitmap(bytes, width, height)
    }

    private fun rgbBytesToBitmap(
            bytes: ByteArray,
            width: Int,
            height: Int,
            pixels: IntArray
    ): Bitmap {
        return io.aatricks.llmedge.vision.ImageUtils.rgbBytesToBitmap(bytes, width, height, pixels)
    }

    private fun determineBatchSize(frameCount: Int): Int =
            when {
                frameCount >= 48 -> MIN_FRAME_BATCH
                frameCount >= 24 -> 6
                else -> MAX_FRAME_BATCH
            }

    private fun warnIfLowMemory(estimatedAdditionalBytes: Long) {
        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory().coerceAtLeast(BYTES_IN_MB)
        val used = runtime.totalMemory() - runtime.freeMemory()
        val projected = used + estimatedAdditionalBytes.coerceAtLeast(0L)
        val ratio = projected.toDouble() / maxMemory.toDouble()
        if (ratio >= MEMORY_PRESSURE_THRESHOLD) {
            Log.w(
                    LOG_TAG,
                    "Memory pressure warning: projected ${(projected / BYTES_IN_MB)} MB of ${(maxMemory / BYTES_IN_MB)} MB heap",
            )
        }
    }

    private fun estimateFrameFootprintBytes(width: Int, height: Int, frameCount: Int): Long {
        val pixels = width.toLong() * height.toLong()
        return pixels * 4L * frameCount
    }

    private fun readNativeMemoryMb(): Long =
            try {
                Debug.getNativeHeapAllocatedSize().coerceAtLeast(0L) / BYTES_IN_MB
            } catch (_: Throwable) {
                val runtime = Runtime.getRuntime()
                (runtime.totalMemory() - runtime.freeMemory()) / BYTES_IN_MB
            }

    private object VideoModelDetector {
        private val VIDEO_KEYWORDS =
                setOf(
                        "wan",
                        "hunyuan",
                        "video",
                        "t2v",
                        "i2v",
                        "ti2v",
                )

        fun isVideoModel(metadata: VideoModelMetadata): Boolean {
            val architecture = metadata.architecture.orEmpty().lowercase(Locale.US)
            if (containsKeyword(architecture)) return true

            val modelType = metadata.modelType.orEmpty().lowercase(Locale.US)
            if (containsKeyword(modelType)) return true

            val filename = metadata.filename.orEmpty().lowercase(Locale.US)
            if (containsKeyword(filename)) return true

            if (metadata.tags.any { containsKeyword(it.lowercase(Locale.US)) }) {
                return true
            }

            return false
        }

        private fun containsKeyword(value: String): Boolean {
            if (value.isEmpty()) return false
            return VIDEO_KEYWORDS.any { keyword -> value.contains(keyword) }
        }
    }
}

object SimpleGenerator {
    suspend fun generate(
            context: Context,
            prompt: String,
            modelId: String = "wan/wan2.1-t2v-1.3B",
            isVideo: Boolean = true,
            outputDir: File = File(context.filesDir, "generations")
    ): File {
        if (!outputDir.exists()) outputDir.mkdirs()

        val sd = StableDiffusion.load(context, modelId = modelId)
        try {
            if (isVideo) {
                val params = StableDiffusion.VideoGenerateParams(prompt = prompt)
                val frames = sd.txt2vid(params)
                val outputFile =
                        File(outputDir, "video_${System.currentTimeMillis()}.png") // Placeholder

                frames.firstOrNull()?.let { bmp ->
                    bmp.compress(Bitmap.CompressFormat.PNG, 100, outputFile.outputStream())
                }
                return outputFile
            } else {
                val params = StableDiffusion.GenerateParams(prompt = prompt)
                val bmp = sd.txt2img(params)
                val outputFile = File(outputDir, "image_${System.currentTimeMillis()}.png")
                bmp.compress(Bitmap.CompressFormat.PNG, 100, outputFile.outputStream())
                return outputFile
            }
        } finally {
            sd.close()
        }
    }
}
