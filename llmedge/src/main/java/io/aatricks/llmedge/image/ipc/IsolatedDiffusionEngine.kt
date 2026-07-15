package io.aatricks.llmedge.image.ipc

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import io.aatricks.llmedge.HangRecoveryPolicy
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.GenerationHangException
import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.core.WorkerCrashedException
import io.aatricks.llmedge.image.GenerationStreamEvent
import io.aatricks.llmedge.image.ImageExecutionPlanner
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.image.VideoGenerationRequest
import io.aatricks.llmedge.image.UpscaleRequest
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageGenerationTraceEvent
import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import java.util.concurrent.CopyOnWriteArraySet
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Binder proxy to the `:llmedge_sd` worker process. Public semantics match
 * [InProcessDiffusionEngine] (single in-flight request, same exceptions for ordinary failures),
 * plus containment: a native crash surfaces as [WorkerCrashedException] (with one automatic CPU
 * retry), and a GPU-driver hang is detected by the [GenerationWatchdog], the worker killed, the
 * verdict persisted, and the request retried on CPU or failed per [HangRecoveryPolicy].
 */
internal class IsolatedDiffusionEngine(
    private val context: Context,
    private val edgeScope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val connectionManager: WorkerConnectionManager = WorkerConnectionManager(context),
    private val verdictStore: BackendVerdictStore = BackendVerdictStore(context),
    private val cpuReaderFor: (Int) -> (() -> Long?) = { pid -> { ProcCpuReader.readCpuTimeMs(pid) } },
    private val clock: () -> Long = { SystemClock.elapsedRealtime() },
) : DiffusionEngine {
    private val generationMutex = Mutex()

    @Volatile private var lastMetrics: GenerationMetrics? = null

    @Volatile private var activeRequest: ActiveRequest? = null

    /** Blacklist entries shipped to every worker (persisted verdicts + session discoveries). */
    private val seedEntries = CopyOnWriteArraySet<Pair<ComputeSubsystem, ComputeBackend>>()

    init {
        if (config.image.persistBackendVerdicts) {
            val persisted = verdictStore.load()
            seedEntries.addAll(persisted)
            BackendRuntimePolicy.seed(persisted)
            if (persisted.isNotEmpty()) {
                AndroidLogAdapter.w(LOG_TAG, "Seeded persisted backend verdicts: $persisted")
            }
        }
        ingestVulkanCreateFailureMarker()
    }

    private class ActiveRequest(
        val connection: WorkerConnectionManager.Connection,
        val isCompleted: () -> Boolean,
    )

    private class ImageRequestPhaseTracker {
        @Volatile var lastPhase: String = DiffusionPhases.REQUESTED
    }

    override suspend fun generate(params: ImageGenerationRequest): Bitmap =
        generateBitmapInternal(params, null)

    private suspend fun generateBitmapInternal(
        params: ImageGenerationRequest,
        onStep: ((Int, Int) -> Unit)?,
    ): Bitmap =
        generationMutex.withLock {
            val phaseTracker = ImageRequestPhaseTracker()
            try {
                val result =
                    runWithRecovery(ComputeSubsystem.IMAGE) { forceCpu ->
                        issueImageRequest(params, forceCpu, phaseTracker, onStep)
                    }
                lastMetrics = result.metrics?.let(IpcCodecs::fromIpc)
                PixelCodec.decodeBitmap(result.frame)
            } catch (oom: io.aatricks.llmedge.core.WorkerKilledByMemoryException) {
                if (isEligibleAutomaticSequentialRetry(params, phaseTracker.lastPhase)) {
                    AndroidLogAdapter.w(LOG_TAG, "Worker OOM-killed while loading; retrying staged image generation")
                    val result =
                        runWithRecovery(ComputeSubsystem.IMAGE) { forceCpu ->
                            issueImageRequest(params.copy(sequential = true), forceCpu, onStep = onStep)
                        }
                    lastMetrics = result.metrics?.let(IpcCodecs::fromIpc)
                    PixelCodec.decodeBitmap(result.frame)
                } else {
                    throw oom
                }
            }
        }

    private fun isEligibleAutomaticSequentialRetry(
        params: ImageGenerationRequest,
        lastPhase: String,
    ): Boolean =
        params.sequential == null &&
            ImageExecutionPlanner.recipeFor(params).supportsSequential &&
            lastPhase == DiffusionPhases.LOADING

    override suspend fun upscale(request: UpscaleRequest): Bitmap =
        generationMutex.withLock {
            val phaseTracker = ImageRequestPhaseTracker()
            val result =
                runWithRecovery(ComputeSubsystem.IMAGE) { forceCpu ->
                    issueUpscaleRequest(request, forceCpu, phaseTracker)
                }
            lastMetrics = result.metrics?.let(IpcCodecs::fromIpc)
            PixelCodec.decodeBitmap(result.frame)
        }

    override fun generateStream(params: ImageGenerationRequest): Flow<GenerationStreamEvent> =
        callbackFlow {
            val job =
                edgeScope.coroutineScope.launch {
                    try {
                        val bitmap = generateBitmapInternal(params) { s, t ->
                            trySend(
                                GenerationStreamEvent.Progress(ProgressEvent.Step("Sampling", s, t))
                            )
                        }
                        trySend(GenerationStreamEvent.Completed(listOf(bitmap)))
                        close()
                    } catch (t: Throwable) {
                        close(t)
                    }
                }
            awaitClose {
                job.cancel()
                cancelGeneration()
            }
        }

    override fun generateVideo(params: VideoGenerationRequest): Flow<GenerationStreamEvent> =
        callbackFlow {
            val producer = this
            val job =
                edgeScope.coroutineScope.launch {
                    try {
                        val result =
                            generationMutex.withLock {
                                runWithRecovery(ComputeSubsystem.VIDEO) { forceCpu ->
                                    issueVideoRequest(params, forceCpu) { message, current, total ->
                                        producer.trySend(
                                            GenerationStreamEvent.Progress(ProgressEvent.Step(message, current, total)),
                                        )
                                    }
                                }
                            }
                        lastMetrics = result.metrics?.let(IpcCodecs::fromIpc)
                        trySend(GenerationStreamEvent.Completed(PixelCodec.decodeFrames(result.frames)))
                        close()
                    } catch (t: Throwable) {
                        close(t)
                    }
                }
            awaitClose {
                job.cancel()
                cancelGeneration()
            }
        }

    override fun cancelGeneration() {
        val request = activeRequest ?: return
        runCatching { request.connection.worker.cancelGeneration() }
        // Native cancel is cooperative (polled between denoise steps); a worker stuck inside a
        // dispatch never observes it. Escalate to SIGKILL after a grace window.
        edgeScope.coroutineScope.launch {
            delay(config.image.watchdog.cancelKillGraceMs)
            if (!request.isCompleted() && !request.connection.dead) {
                connectionManager.killWorker(request.connection)
            }
        }
    }

    override fun lastGenerationMetrics(): GenerationMetrics? = lastMetrics

    /** Trace events stay inside the worker process; not available in isolated mode. */
    override fun lastImageRequestTraceForTests(): List<ImageGenerationTraceEvent> = emptyList()

    override fun close() {
        val request = activeRequest
        if (request != null && !request.connection.dead) {
            runCatching { request.connection.worker.cancelGeneration() }
        }
        // Synchronous like ManagedRuntimeBase.closeOnce: the owning scope tears down right after.
        kotlinx.coroutines.runBlocking { connectionManager.close() }
    }

    /** Debug/test hook: forwards fault-injection args to the (freshly connected) worker. */
    internal suspend fun installFaultInjectionForTests(args: android.os.Bundle) {
        val connection = connectionManager.connect(buildInitConfig(forceCpu = false))
        connection.worker.installFaultInjection(args)
    }

    private suspend fun <T> runWithRecovery(
        subsystem: ComputeSubsystem,
        issue: suspend (forceCpu: Boolean) -> T,
    ): T {
        ingestVulkanCreateFailureMarker()
        return try {
            issue(false)
        } catch (hang: GenerationHangException) {
            recordHangVerdict(subsystem, hang.backend)
            when (config.image.hangRecoveryPolicy) {
                HangRecoveryPolicy.FAIL_FAST -> throw hang
                HangRecoveryPolicy.RETRY_CPU_THEN_FAIL -> {
                    AndroidLogAdapter.w(LOG_TAG, "Worker hang (${hang.message}); retrying on CPU")
                    issue(true)
                }
            }
        } catch (crash: WorkerCrashedException) {
            // A crash whose tombstone names a vulkan driver library taints the whole worker
            // process — even when the session computed on CPU (the capacity probe or a failed
            // Vulkan create attempt initialized the driver). Quarantine Vulkan and retry once in
            // a fresh worker that never loads the driver.
            val vulkanImplicated = crashImplicatesVulkanDriver(crash.crashSummary)
            if (vulkanImplicated) {
                recordHangVerdict(subsystem, ComputeBackend.VULKAN.name)
            }
            if (crash.backend == ComputeBackend.CPU.name) {
                if (!vulkanImplicated) throw crash
                AndroidLogAdapter.w(
                    LOG_TAG,
                    "CPU-session crash implicates the Vulkan driver (${crash.message}); retrying with Vulkan quarantined",
                )
                issue(true)
            } else {
                markSessionBlacklist(subsystem, crash.backend)
                AndroidLogAdapter.w(LOG_TAG, "Worker crashed (${crash.message}); retrying on CPU")
                issue(true)
            }
        }
    }

    /** True when the tombstone summary names a vulkan driver library (e.g. vulkan.mtk.so, libvulkan.so). */
    private fun crashImplicatesVulkanDriver(crashSummary: String?): Boolean =
        crashSummary != null && VULKAN_DRIVER_LIBRARY.containsMatchIn(crashSummary)

    /** Converts a worker-side "Vulkan create failed" breadcrumb into a persisted verdict. */
    private fun ingestVulkanCreateFailureMarker() {
        if (!io.aatricks.llmedge.image.diffusion.VulkanCreateFailureMarker.consume(context)) return
        AndroidLogAdapter.w(LOG_TAG, "Worker reported a failed Vulkan create attempt; quarantining Vulkan")
        recordHangVerdict(ComputeSubsystem.IMAGE, ComputeBackend.VULKAN.name)
        recordHangVerdict(ComputeSubsystem.VIDEO, ComputeBackend.VULKAN.name)
    }

    private suspend fun issueImageRequest(
        params: ImageGenerationRequest,
        forceCpu: Boolean,
        phaseTracker: ImageRequestPhaseTracker? = null,
        onStep: ((Int, Int) -> Unit)? = null,
    ): IpcImageResult {
        val deferred = CompletableDeferred<IpcImageResult>()
        return runRequest(deferred, forceCpu) { connection, watchdog ->
            val callback =
                object : IDiffusionResultCallback.Stub() {
                    override fun onPhase(update: PhaseUpdate) {
                        phaseTracker?.lastPhase = update.phase
                        dispatchPhase(watchdog, update)
                        if (update.phase == DiffusionPhases.STEP) {
                            onStep?.invoke(update.step, update.totalSteps)
                        }
                    }

                    override fun onCompleted(result: IpcImageResult) {
                        deferred.complete(result)
                    }

                    override fun onFailed(failure: IpcFailure) {
                        deferred.completeExceptionally(mapFailure("image generation", failure))
                    }
                }
            connection.worker.generateImage(IpcCodecs.toIpc(params), callback)
        }
    }

    private suspend fun issueUpscaleRequest(
        request: UpscaleRequest,
        forceCpu: Boolean,
        phaseTracker: ImageRequestPhaseTracker? = null,
    ): IpcImageResult {
        val deferred = CompletableDeferred<IpcImageResult>()
        val finalRequest = if (forceCpu) request.copy(useVulkan = false) else request
        val ipcRequest = IpcCodecs.toIpc(finalRequest)
        try {
            return runRequest(deferred, forceCpu) { connection, watchdog ->
                val callback =
                    object : IDiffusionResultCallback.Stub() {
                        override fun onPhase(update: PhaseUpdate) {
                            phaseTracker?.lastPhase = update.phase
                            dispatchPhase(watchdog, update)
                        }

                        override fun onCompleted(result: IpcImageResult) {
                            deferred.complete(result)
                        }

                        override fun onFailed(failure: IpcFailure) {
                            deferred.completeExceptionally(mapFailure("upscale", failure))
                        }
                    }
                connection.worker.upscaleImage(ipcRequest, callback)
            }
        } finally {
            ipcRequest.input.memory.close()
        }
    }

    private suspend fun issueVideoRequest(
        params: VideoGenerationRequest,
        forceCpu: Boolean,
        onProgress: (String, Int, Int) -> Unit,
    ): IpcVideoResult {
        val deferred = CompletableDeferred<IpcVideoResult>()
        val ipcRequest = IpcCodecs.toIpc(params)
        try {
            return runRequest(deferred, forceCpu) { connection, watchdog ->
                val callback =
                    object : IDiffusionVideoCallback.Stub() {
                        override fun onPhase(update: PhaseUpdate) = dispatchPhase(watchdog, update)

                        override fun onProgress(
                            message: String,
                            current: Int,
                            total: Int,
                        ) {
                            watchdog?.onStep(current, total)
                            onProgress(message, current, total)
                        }

                        override fun onCompleted(result: IpcVideoResult) {
                            deferred.complete(result)
                        }

                        override fun onFailed(failure: IpcFailure) {
                            deferred.completeExceptionally(mapFailure("video generation", failure))
                        }
                    }
                connection.worker.generateVideo(ipcRequest, callback)
            }
        } finally {
            // Binder dups the fd on send; the host-side initImage buffer is done either way.
            ipcRequest.initImage?.memory?.close()
        }
    }

    /**
     * Shared request lifecycle: connect, arm death handling + watchdog, send the request via
     * [send], await the result, and tear everything down inline. Coroutine cancellation sends a
     * cooperative cancel to the worker and escalates to a kill after the grace window.
     */
    private suspend fun <T> runRequest(
        deferred: CompletableDeferred<T>,
        forceCpu: Boolean,
        send: (WorkerConnectionManager.Connection, GenerationWatchdog?) -> Unit,
    ): T {
        val connection = connectionManager.connect(buildInitConfig(forceCpu))
        val killedByWatchdog = AtomicBoolean(false)
        val verdictStallMs = AtomicLong(0L)

        var watchdog: GenerationWatchdog? = null
        if (config.image.watchdog.enabled) {
            watchdog =
                GenerationWatchdog(
                    config = config.image.watchdog,
                    clock = clock,
                    cpuTimeMsReader = cpuReaderFor(connection.pid),
                ) { _, _, stallMs ->
                    killedByWatchdog.set(true)
                    verdictStallMs.set(stallMs)
                    connectionManager.killWorker(connection)
                }
        }
        val watchdogRef = watchdog

        connection.onDeath = {
            deferred.completeExceptionally(
                WorkerFailureClassifier.classify(
                    context = context,
                    pid = connection.pid,
                    lastPhase = watchdogRef?.lastPhase() ?: DiffusionPhases.REQUESTED,
                    lastBackend = watchdogRef?.lastBackend(),
                    killedByWatchdog = killedByWatchdog.get(),
                    stallMs = verdictStallMs.get(),
                ),
            )
        }
        if (connection.dead) {
            // Died between connect and arming the handler; classify immediately.
            connection.onDeath?.invoke()
        }

        activeRequest = ActiveRequest(connection) { deferred.isCompleted }

        val ticker =
            watchdog?.let { armed ->
                edgeScope.coroutineScope.launch {
                    while (isActive && !deferred.isCompleted) {
                        delay(config.image.watchdog.cpuSampleIntervalMs)
                        armed.sample()
                    }
                }
            }

        try {
            if (!deferred.isCompleted) {
                try {
                    send(connection, watchdog)
                } catch (remote: Throwable) {
                    // The oneway send itself failed (e.g. DeadObjectException between connect and send).
                    if (!deferred.isCompleted) {
                        deferred.completeExceptionally(WorkerCrashedException(backend = null, exitReason = null))
                    }
                }
            }
            return deferred.await()
        } catch (cancelled: CancellationException) {
            runCatching { connection.worker.cancelGeneration() }
            edgeScope.coroutineScope.launch {
                delay(config.image.watchdog.cancelKillGraceMs)
                if (!deferred.isCompleted && !connection.dead) {
                    connectionManager.killWorker(connection)
                }
            }
            throw cancelled
        } finally {
            withContext(NonCancellable) {
                ticker?.cancel()
                watchdog?.stop()
                connection.onDeath = null
                activeRequest = null
            }
        }
    }

    private fun dispatchPhase(
        watchdog: GenerationWatchdog?,
        update: PhaseUpdate,
    ) {
        if (update.phase == DiffusionPhases.STEP) {
            watchdog?.onStep(update.step, update.totalSteps)
        } else {
            watchdog?.onPhase(update.phase, update.backend)
        }
    }

    private fun buildInitConfig(forceCpu: Boolean): WorkerInitConfig =
        WorkerInitConfig(
            cacheMaxEntries = config.image.cache.maxEntries,
            cacheMaxMemoryMb = config.image.cache.maxMemoryMb,
            preferPerformanceMode = config.image.preferPerformanceMode,
            useVulkan = config.image.useVulkan && !forceCpu,
            blacklistSeed = seedEntries.map { (subsystem, backend) -> "${subsystem.name}:${backend.name}" },
        )

    private fun recordHangVerdict(
        subsystem: ComputeSubsystem,
        backendName: String?,
    ) {
        val backend = backendName?.let { name -> ComputeBackend.entries.firstOrNull { it.name == name } }
        // An unknown backend means the hang happened before any phase heartbeat (load or
        // conditioner dispatch). GPU is the only credible culprit for that signature.
        val verdictBackend = backend ?: ComputeBackend.VULKAN
        if (verdictBackend == ComputeBackend.CPU) return
        seedEntries.add(subsystem to verdictBackend)
        BackendRuntimePolicy.seed(listOf(subsystem to verdictBackend))
        if (config.image.persistBackendVerdicts) {
            verdictStore.recordHang(subsystem, verdictBackend)
        }
    }

    private fun markSessionBlacklist(
        subsystem: ComputeSubsystem,
        backendName: String?,
    ) {
        val backend = backendName?.let { name -> ComputeBackend.entries.firstOrNull { it.name == name } } ?: return
        if (backend == ComputeBackend.CPU) return
        seedEntries.add(subsystem to backend)
        BackendRuntimePolicy.seed(listOf(subsystem to backend))
    }

    private fun mapFailure(
        operation: String,
        failure: IpcFailure,
    ): Throwable =
        when (failure.code) {
            IpcFailure.CODE_CANCELLED -> CancellationException("Generation cancelled in worker")
            else ->
                InferenceFailedException(
                    operation,
                    "${failure.exceptionClass}: ${failure.message ?: "no message"} (worker process)",
                )
        }

    companion object {
        private const val LOG_TAG = "IsolatedDiffusion"
        private val VULKAN_DRIVER_LIBRARY = Regex("(?i)(lib)?vulkan[\\w.]*\\.so")
    }
}
