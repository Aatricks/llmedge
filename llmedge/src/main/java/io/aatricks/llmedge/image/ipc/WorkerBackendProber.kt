package io.aatricks.llmedge.image.ipc

import android.content.Context
import io.aatricks.llmedge.ComputeBackendAvailability
import io.aatricks.llmedge.VulkanDeviceInfo
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.WorkerKilledByMemoryException
import io.aatricks.llmedge.runtime.BackendRuntimePolicy
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

internal object WorkerBackendProber {
    private const val LOG_TAG = "WorkerBackendProber"
    private const val PROBE_TIMEOUT_MS = 10_000L

    @Volatile private var cached: ComputeBackendAvailability? = null
    @Volatile private var vulkanQuarantined: Boolean = false
    private val mutex = Mutex()

    fun cachedOrNull(): ComputeBackendAvailability? = cached

    fun persistedOrNull(context: Context): ComputeBackendAvailability? {
        val store = BackendVerdictStore(context)
        val hasVulkanVerdict = store.load().any { it.second == ComputeBackend.VULKAN }
        if (hasVulkanVerdict) {
            vulkanQuarantined = true
        }
        val persisted = store.loadImageProbe() ?: return null
        return if (hasVulkanVerdict) {
            ComputeBackendAvailability(persisted.openClAvailable, false, null)
        } else {
            persisted
        }
    }

    internal fun isVulkanQuarantined(): Boolean = vulkanQuarantined

    internal fun reset() {
        cached = null
        vulkanQuarantined = false
    }

    suspend fun probe(context: Context): ComputeBackendAvailability = mutex.withLock {
        cached?.let { return it }

        val store = BackendVerdictStore(context)
        val loadedVerdicts = store.load()
        val hasVulkanVerdict = loadedVerdicts.any { it.second == ComputeBackend.VULKAN }
        if (hasVulkanVerdict) {
            vulkanQuarantined = true
        }
        
        val persisted = store.loadImageProbe()
        if (persisted != null) {
            if (hasVulkanVerdict) {
                val updated = ComputeBackendAvailability(persisted.openClAvailable, false, null)
                cached = updated
                return updated
            }
            cached = persisted
            return persisted
        }

        val blacklistSeed = if (hasVulkanVerdict) {
            listOf("IMAGE:VULKAN")
        } else emptyList()

        return executeProbe(context, blacklistSeed, hasVulkanVerdict)
    }

    private suspend fun executeProbe(context: Context, blacklistSeed: List<String>, alreadyHasVulkanVerdict: Boolean): ComputeBackendAvailability {
        val manager = WorkerConnectionManager(context)
        val store = BackendVerdictStore(context)
        var connection: WorkerConnectionManager.Connection? = null
        try {
            // Timeout covers the bind as well as the binder call: a stalled service spawn
            // must fail the probe, not park it forever. It runs on the IO dispatcher's real
            // clock so runTest virtual time cannot fire it early.
            val result = withContext(Dispatchers.IO) {
                withTimeout(PROBE_TIMEOUT_MS) {
                    val live = manager.connect(null)
                    connection = live
                    live.worker.probeBackends(blacklistSeed)
                }
            }
            val availability = mapResult(result)
            store.recordImageProbe(availability)
            cached = availability
            return availability
        } catch (t: Throwable) {
            if (t is kotlinx.coroutines.CancellationException && t !is kotlinx.coroutines.TimeoutCancellationException) {
                throw t
            }
            if (t is kotlinx.coroutines.TimeoutCancellationException) {
                connection?.let { manager.killWorker(it) }
            }
            
            val classifierEx = connection?.let {
                WorkerFailureClassifier.classify(
                    context = context,
                    pid = it.pid,
                    lastPhase = "PROBE",
                    lastBackend = "VULKAN",
                    killedByWatchdog = t is kotlinx.coroutines.TimeoutCancellationException,
                    stallMs = PROBE_TIMEOUT_MS
                )
            }
            AndroidLogAdapter.w(
                LOG_TAG,
                "Backend probe failed: ${t.javaClass.simpleName} (connection=${connection != null}, " +
                    "classified=${classifierEx?.javaClass?.simpleName}, priorVerdict=$alreadyHasVulkanVerdict)",
            )

            if (connection == null) {
                return ComputeBackendAvailability(false, false, null)
            }

            if (classifierEx is WorkerKilledByMemoryException) {
                return ComputeBackendAvailability(false, false, null)
            }
            
            if (!alreadyHasVulkanVerdict) {
                store.recordHang(ComputeSubsystem.IMAGE, ComputeBackend.VULKAN)
                store.recordHang(ComputeSubsystem.VIDEO, ComputeBackend.VULKAN)
                BackendRuntimePolicy.seed(listOf(ComputeSubsystem.IMAGE to ComputeBackend.VULKAN, ComputeSubsystem.VIDEO to ComputeBackend.VULKAN))
                vulkanQuarantined = true
                
                // Retry ONCE with Vulkan blacklisted
                return executeRetryProbe(context, manager, store, listOf("IMAGE:VULKAN"))
            } else {
                return ComputeBackendAvailability(false, false, null)
            }
        } finally {
            manager.close()
        }
    }
    
    private suspend fun executeRetryProbe(context: Context, manager: WorkerConnectionManager, store: BackendVerdictStore, blacklistSeed: List<String>): ComputeBackendAvailability {
        var connection: WorkerConnectionManager.Connection? = null
        try {
            val result = withContext(Dispatchers.IO) {
                withTimeout(PROBE_TIMEOUT_MS) {
                    val live = manager.connect(null)
                    connection = live
                    live.worker.probeBackends(blacklistSeed)
                }
            }
            val mapped = mapResult(result)
            val availability = ComputeBackendAvailability(mapped.openClAvailable, false, null)
            store.recordImageProbe(availability)
            cached = availability
            return availability
        } catch (t: Throwable) {
            if (t is kotlinx.coroutines.CancellationException && t !is kotlinx.coroutines.TimeoutCancellationException) {
                throw t
            }
            if (t is kotlinx.coroutines.TimeoutCancellationException) {
                connection?.let { manager.killWorker(it) }
            }
            return ComputeBackendAvailability(false, false, null)
        }
    }

    private fun mapResult(result: IpcBackendProbeResult): ComputeBackendAvailability {
        val vulkanAvailable = result.vulkanDeviceCount > 0 && result.vulkanTotalBytes > 0
        val deviceInfo = if (vulkanAvailable) {
            VulkanDeviceInfo(
                deviceCount = result.vulkanDeviceCount,
                freeMemoryMB = result.vulkanFreeBytes / (1024 * 1024),
                totalMemoryMB = result.vulkanTotalBytes / (1024 * 1024),
                deviceIndex = 0
            )
        } else null
        return ComputeBackendAvailability(
            openClAvailable = result.openClAvailable,
            vulkanAvailable = vulkanAvailable,
            vulkanDeviceInfo = deviceInfo
        )
    }
}
