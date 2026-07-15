package io.aatricks.llmedge.image.ipc

import android.app.Service
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Bundle
import android.os.IBinder
import android.os.Process
import android.os.SystemClock
import io.aatricks.llmedge.ImageRuntimeConfig
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.RuntimeCacheConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.image.DiffusionPhaseListener
import io.aatricks.llmedge.image.GenerationStreamEvent
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Bound service hosting the diffusion stack in the `:llmedge_sd` process. It simply runs an
 * [InProcessDiffusionEngine] here — same code, different process — and forwards phase heartbeats
 * so the host-side watchdog can tell a hung worker from a busy one. Dying (crash, LMK, watchdog
 * kill) is part of its contract: the host observes binderDied and recovers.
 */
internal class DiffusionWorkerService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private var bootstrap: ClientBootstrapContext? = null
    private var engine: InProcessDiffusionEngine? = null
    private var engineConfig: WorkerInitConfig? = null

    /** Phase sink for the request currently generating (single in-flight). */
    @Volatile private var phaseSink: ((PhaseUpdate) -> Unit)? = null

    @Volatile private var faultInjection: Bundle? = null

    private val phaseRelay =
        object : DiffusionPhaseListener {
            override fun onPhase(
                phase: String,
                backend: String?,
            ) {
                phaseSink?.invoke(PhaseUpdate(phase, backend, 0, 0, SystemClock.uptimeMillis()))
            }

            override fun onStep(
                step: Int,
                totalSteps: Int,
            ) {
                phaseSink?.invoke(PhaseUpdate(DiffusionPhases.STEP, null, step, totalSteps, SystemClock.uptimeMillis()))
            }
        }

    private val binder =
        object : IDiffusionWorker.Stub() {
            override fun getPid(): Int = Process.myPid()

            override fun initialize(config: WorkerInitConfig) {
                synchronized(this@DiffusionWorkerService) {
                    if (config == engineConfig && engine != null) return
                    engine?.close()
                    bootstrap?.close()
                    val newBootstrap =
                        ClientBootstrapContext.create(
                            context = applicationContext,
                            scope = serviceScope,
                            inferenceThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
                        )
                    BackendRuntimePolicy.seed(
                        config.blacklistSeed.mapNotNull { entry ->
                            val parts = entry.split(':')
                            if (parts.size != 2) return@mapNotNull null
                            val subsystem = ComputeSubsystem.entries.firstOrNull { it.name == parts[0] } ?: return@mapNotNull null
                            val backend = ComputeBackend.entries.firstOrNull { it.name == parts[1] } ?: return@mapNotNull null
                            subsystem to backend
                        },
                    )
                    // Must run before the first native call: keeps unstable vendor Vulkan drivers
                    // from ever loading in this process (probe included) once Vulkan is disallowed.
                    WorkerVulkanGate.apply(
                        WorkerVulkanGate.shouldDisable(config.useVulkan, config.blacklistSeed),
                    )
                    bootstrap = newBootstrap
                    engine =
                        InProcessDiffusionEngine(
                            appContext = applicationContext,
                            edgeScope = newBootstrap.edgeScope,
                            config =
                                LLMEdgeConfig(
                                    image =
                                        ImageRuntimeConfig(
                                            cache =
                                                RuntimeCacheConfig(
                                                    maxEntries = config.cacheMaxEntries,
                                                    maxMemoryMb = config.cacheMaxMemoryMb,
                                                ),
                                            preferPerformanceMode = config.preferPerformanceMode,
                                            useVulkan = config.useVulkan,
                                        ),
                                ),
                            modelRepository = DefaultModelRepository(),
                            logTag = LOG_TAG,
                            phaseListener = phaseRelay,
                        )
                    engineConfig = config
                    AndroidLogAdapter.i(LOG_TAG, "Worker engine initialized (useVulkan=${config.useVulkan}, seed=${config.blacklistSeed})")
                }
            }

            override fun generateImage(
                request: IpcImageRequest,
                callback: IDiffusionResultCallback,
            ) {
                val activeEngine = engine
                serviceScope.launch {
                    phaseSink = { update -> runCatching { callback.onPhase(update) } }
                    if (simulateFaultIfRequested()) return@launch
                    if (activeEngine == null) {
                        runCatching { callback.onFailed(failure(IllegalStateException("worker not initialized"))) }
                        phaseSink = null
                        return@launch
                    }
                    try {
                        val bitmap = activeEngine.generate(IpcCodecs.fromIpc(request))
                        val metrics = activeEngine.lastGenerationMetrics()?.let(IpcCodecs::toIpc)
                        val frame = PixelCodec.encodeBitmap(bitmap, "llmedge_image_result")
                        try {
                            callback.onCompleted(IpcImageResult(frame = frame, metrics = metrics))
                        } finally {
                            // Binder dups the fd during the call; our SharedMemory object is done.
                            frame.memory.close()
                        }
                    } catch (t: Throwable) {
                        runCatching { callback.onFailed(failure(t)) }
                    } finally {
                        phaseSink = null
                    }
                }
            }

            override fun generateVideo(
                request: IpcVideoRequest,
                callback: IDiffusionVideoCallback,
            ) {
                val activeEngine = engine
                serviceScope.launch {
                    phaseSink = { update -> runCatching { callback.onPhase(update) } }
                    if (simulateFaultIfRequested()) return@launch
                    if (activeEngine == null) {
                        runCatching { callback.onFailed(failure(IllegalStateException("worker not initialized"))) }
                        phaseSink = null
                        return@launch
                    }
                    try {
                        var completed = false
                        activeEngine.generateVideo(IpcCodecs.fromIpc(request)).collect { event ->
                            when (event) {
                                is GenerationStreamEvent.Progress ->
                                    runCatching {
                                        callback.onProgress(event.update.message, event.update.current, event.update.total)
                                    }
                                is GenerationStreamEvent.Completed -> {
                                    completed = true
                                    val metrics = activeEngine.lastGenerationMetrics()?.let(IpcCodecs::toIpc)
                                    val frames = PixelCodec.encodeFrames(event.frames, "llmedge_video_result")
                                    try {
                                        callback.onCompleted(IpcVideoResult(frames = frames, metrics = metrics))
                                    } finally {
                                        frames.memory.close()
                                    }
                                }
                            }
                        }
                        if (!completed) {
                            runCatching { callback.onFailed(failure(IllegalStateException("video flow ended without frames"))) }
                        }
                    } catch (t: Throwable) {
                        runCatching { callback.onFailed(failure(t)) }
                    } finally {
                        phaseSink = null
                    }
                }
            }

            override fun cancelGeneration() {
                engine?.cancelGeneration()
            }

            override fun installFaultInjection(args: Bundle) {
                val debuggable = (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
                if (!debuggable) throw SecurityException("Fault injection requires a debuggable build")
                faultInjection = args
                AndroidLogAdapter.w(LOG_TAG, "Fault injection installed: ${args.getString(FAULT_MODE)}")
            }

            override fun upscaleImage(
                request: IpcUpscaleRequest,
                callback: IDiffusionResultCallback,
            ) {
                val activeEngine = engine
                serviceScope.launch {
                    phaseSink = { update -> runCatching { callback.onPhase(update) } }
                    if (simulateFaultIfRequested()) return@launch
                    if (activeEngine == null) {
                        runCatching { callback.onFailed(failure(IllegalStateException("worker not initialized"))) }
                        phaseSink = null
                        runCatching { request.input.memory.close() }
                        return@launch
                    }
                    try {
                        phaseRelay.onPhase(DiffusionPhases.LOADING, null)
                        val kotlinRequest = IpcCodecs.fromIpc(request)
                        val useVulkan = request.useVulkan && !BackendRuntimePolicy.isBlacklisted(ComputeSubsystem.IMAGE, ComputeBackend.VULKAN)
                        val backendName = if (useVulkan) "vulkan" else "cpu"
                        phaseRelay.onPhase(DiffusionPhases.GENERATING, backendName)
                        val bitmap = activeEngine.upscale(kotlinRequest)
                        val frame = PixelCodec.encodeBitmap(bitmap, "llmedge_upscale_result")
                        try {
                            callback.onCompleted(IpcImageResult(frame = frame, metrics = null))
                        } finally {
                            frame.memory.close()
                        }
                    } catch (t: Throwable) {
                        runCatching { callback.onFailed(failure(t)) }
                    } finally {
                        phaseSink = null
                        runCatching { request.input.memory.close() }
                    }
                }
            }
        }

    /** Returns true when a fault was simulated instead of generating. */
    private fun simulateFaultIfRequested(): Boolean {
        val args = faultInjection ?: return false
        when (args.getString(FAULT_MODE)) {
            FAULT_HANG -> {
                // Reproduce the driver-deadlock signature: no callbacks, thread parked, 0% CPU.
                AndroidLogAdapter.w(LOG_TAG, "Simulating dispatch hang")
                java.util.concurrent.CountDownLatch(1).await()
            }
            FAULT_HANG_AFTER_PHASE -> {
                phaseSink?.invoke(
                    PhaseUpdate(DiffusionPhases.LOADING, args.getString(FAULT_BACKEND) ?: "VULKAN", 0, 0, SystemClock.uptimeMillis()),
                )
                AndroidLogAdapter.w(LOG_TAG, "Simulating dispatch hang after LOADING phase")
                java.util.concurrent.CountDownLatch(1).await()
            }
            FAULT_CRASH -> {
                AndroidLogAdapter.w(LOG_TAG, "Simulating native crash")
                Process.killProcess(Process.myPid())
            }
            FAULT_JAVA_CRASH -> {
                // A real uncaught JVM exception (REASON_CRASH): thrown on a fresh thread so it
                // escapes every try/catch, exercising the uncaught-handler breadcrumb path.
                AndroidLogAdapter.w(LOG_TAG, "Simulating uncaught JVM crash")
                Thread { throw IllegalStateException("Injected uncaught JVM fault") }.start()
                java.util.concurrent.CountDownLatch(1).await()
            }
            FAULT_NATIVE_ABORT -> {
                // A real SIGABRT (REASON_CRASH_NATIVE): debuggerd writes an actual tombstone,
                // exercising the ApplicationExitInfo trace → NativeTombstoneSummary path.
                AndroidLogAdapter.w(LOG_TAG, "Simulating native abort")
                android.system.Os.kill(Process.myPid(), android.system.OsConstants.SIGABRT)
            }
            else -> return false
        }
        return true
    }

    private fun failure(t: Throwable): IpcFailure =
        IpcFailure(
            code =
                if (t is kotlinx.coroutines.CancellationException) {
                    IpcFailure.CODE_CANCELLED
                } else {
                    IpcFailure.CODE_GENERIC
                },
            exceptionClass = t.javaClass.name,
            message = t.message,
            backend = null,
        )

    override fun onCreate() {
        super.onCreate()
        // A generation exception is caught and reported over binder; an *uncaught* one kills this
        // process (exitReason REASON_CRASH) with the stack lost to logcat. Persist it so the host
        // can surface it in WorkerCrashedException — the only way to see it without adb.
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            runCatching {
                WorkerFailureClassifier.crashBreadcrumbFile(this, Process.myPid())
                    .writeText("thread=${thread.name}\n${throwable.stackTraceToString()}".take(4000))
            }
            previous?.uncaughtException(thread, throwable)
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        engine?.close()
        bootstrap?.close()
        serviceScope.cancel()
        super.onDestroy()
    }

    companion object {
        private const val LOG_TAG = "DiffusionWorker"
        const val FAULT_MODE = "mode"
        const val FAULT_BACKEND = "backend"
        const val FAULT_HANG = "hang"
        const val FAULT_HANG_AFTER_PHASE = "hang-after-phase"
        const val FAULT_CRASH = "crash"
        const val FAULT_JAVA_CRASH = "java-crash"
        const val FAULT_NATIVE_ABORT = "native-abort"
    }
}
