package io.aatricks.llmedge.image.diffusion

import android.content.Context
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.NativeBridgeProvider
import io.aatricks.llmedge.core.NativeLibraryLoader
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher

internal object StableDiffusionCompanionSupport {
    private const val LOG_TAG = "StableDiffusion"
    private const val DIFFUSION_DISPATCHER_THREADS = 2

    private val diffusionWorkerIds = AtomicInteger(0)

    val diffusionDispatcher: CoroutineDispatcher =
        Executors.newFixedThreadPool(DIFFUSION_DISPATCHER_THREADS) {
            val workerId = diffusionWorkerIds.incrementAndGet()
            Thread(it, "llmedge-diffusion-$workerId").apply { isDaemon = true }
        }.asCoroutineDispatcher()

    @Volatile
    private var isNativeLibraryAvailable: Boolean = false

    private var nativeBridgeOverriddenForTests: Boolean = false

    private val defaultNativeLibrarySupport =
        StableDiffusionNativeLibrarySupport {
            val disableNativeLoad = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")
            isNativeLibraryAvailable = !disableNativeLoad
            if (disableNativeLoad) {
                AndroidLogAdapter.i(LOG_TAG, "Native load disabled via llmedge.disableNativeLoad=true")
            } else {
                NativeLibraryLoader.ensureStableDiffusionLoaded(
                    required = true,
                    onDebug = { message -> AndroidLogAdapter.d(LOG_TAG, message) },
                    onError = { message, throwable -> AndroidLogAdapter.e(LOG_TAG, message, throwable) },
                    verifyBindings = null,
                )
            }
        }

    private val noOpNativeLibrarySupport = StableDiffusionNativeLibrarySupport { }

    @Volatile
    private var nativeLibrarySupportOverride: StableDiffusionNativeLibrarySupport? = null

    private val nativeBridgeProvider =
        NativeBridgeProvider(StableDiffusionNativeBridgeSupport.defaultProvider())

    fun computeEffectiveSequentialLoad(
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

    fun createNativeBridge(instance: StableDiffusion): StableDiffusion.NativeBridge =
        nativeBridgeProvider.create(instance)

    fun currentNativeLibrarySupport(): StableDiffusionNativeLibrarySupport =
        nativeLibrarySupportOverride ?: defaultNativeLibrarySupport

    fun isNativeLibraryLoaded(checkBindings: () -> Boolean): Boolean =
        try {
            checkBindings()
        } catch (_: Throwable) {
            false
        }

    fun enableNativeBridgeForTests() {
        if (!isNativeLibraryAvailable) {
            isNativeLibraryAvailable = true
        }
    }

    fun overrideNativeBridgeForTests(provider: (StableDiffusion) -> StableDiffusion.NativeBridge) {
        nativeBridgeProvider.override(provider)
        nativeBridgeOverriddenForTests = true
        nativeLibrarySupportOverride = noOpNativeLibrarySupport
    }

    fun resetNativeBridgeForTests() {
        nativeBridgeProvider.reset()
        nativeBridgeOverriddenForTests = false
        nativeLibrarySupportOverride = null
    }

    fun getVulkanDeviceCount(nativeCall: () -> Int): Int =
        try {
            currentNativeLibrarySupport().ensureLoaded()
            nativeCall()
        } catch (_: Throwable) {
            0
        }

    fun getVulkanDeviceMemory(nativeCall: () -> LongArray?): LongArray? =
        try {
            currentNativeLibrarySupport().ensureLoaded()
            nativeCall()
        } catch (_: Throwable) {
            null
        }

    fun getVulkanDeviceDescription(nativeCall: () -> String?): String? =
        try {
            currentNativeLibrarySupport().ensureLoaded()
            nativeCall()
        } catch (_: Throwable) {
            null
        }

    fun estimateModelParamsMemoryBytes(nativeCall: () -> Long): Long =
        try {
            currentNativeLibrarySupport().ensureLoaded()
            nativeCall()
        } catch (_: Throwable) {
            0L
        }

    fun checkBindings(nativeCall: () -> Boolean): Boolean =
        try {
            currentNativeLibrarySupport().ensureLoaded()
            nativeCall()
        } catch (_: Throwable) {
            false
        }

    fun isOpenClAvailable(nativeCall: () -> Boolean): Boolean =
        try {
            currentNativeLibrarySupport().ensureLoaded()
            nativeCall()
        } catch (_: Throwable) {
            false
        }

    fun supportIsNativeLibraryAvailable(): Boolean = isNativeLibraryAvailable

    fun supportNativeBridgeOverriddenForTests(): Boolean = nativeBridgeOverriddenForTests

    fun logWarning(message: String) {
        AndroidLogAdapter.w(LOG_TAG, message)
    }
}
