package io.aatricks.llmedge.image.ipc

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.os.IBinder
import android.os.Process
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import io.aatricks.llmedge.DiffusionWorkerMode
import io.aatricks.llmedge.HangRecoveryPolicy
import io.aatricks.llmedge.ImageRuntimeConfig
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.WorkerWatchdogConfig
import io.aatricks.llmedge.core.GenerationHangException
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.runtime.ComputeBackend
import io.aatricks.llmedge.runtime.ComputeSubsystem
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class IsolatedWorkerTest {
    private lateinit var context: Context
    private lateinit var parentScope: CoroutineScope
    private lateinit var edgeScope: LLMEdgeScope

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        parentScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        edgeScope = LLMEdgeScope(parentScope, 2)
        BackendVerdictStore(context).reset()
    }

    @After
    fun tearDown() {
        BackendVerdictStore(context).reset()
        edgeScope.close()
        parentScope.cancel()
    }

    @Test
    fun sharedMemoryPixelCodecRoundTrips() {
        val bitmap = Bitmap.createBitmap(16, 8, Bitmap.Config.ARGB_8888)
        for (x in 0 until 16) for (y in 0 until 8) {
            bitmap.setPixel(x, y, Color.rgb(x * 16, y * 32, (x + y) * 8))
        }
        val decoded = PixelCodec.decodeBitmap(PixelCodec.encodeBitmap(bitmap, "test_frame"))
        assertEquals(bitmap.width, decoded.width)
        assertEquals(bitmap.height, decoded.height)
        assertEquals(bitmap.getPixel(3, 5), decoded.getPixel(3, 5))
        assertEquals(bitmap.getPixel(15, 7), decoded.getPixel(15, 7))
    }

    @Test
    fun workerServiceBindsInSeparateProcess() {
        val latch = CountDownLatch(1)
        var binder: IBinder? = null
        val connection =
            object : ServiceConnection {
                override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                    binder = service
                    latch.countDown()
                }

                override fun onServiceDisconnected(name: ComponentName?) = Unit
            }
        try {
            assertTrue(
                context.bindService(
                    Intent(context, DiffusionWorkerService::class.java),
                    connection,
                    Context.BIND_AUTO_CREATE,
                ),
            )
            assertTrue("service did not connect", latch.await(20, TimeUnit.SECONDS))
            val worker = IDiffusionWorker.Stub.asInterface(binder)
            val workerPid = worker.pid
            assertNotEquals("worker must run in its own process", Process.myPid(), workerPid)
            // initialize is idempotent
            val config =
                WorkerInitConfig(
                    cacheMaxEntries = 1,
                    cacheMaxMemoryMb = 1024,
                    preferPerformanceMode = false,
                    useVulkan = false,
                    blacklistSeed = emptyList(),
                )
            worker.initialize(config)
            worker.initialize(config)
        } finally {
            runCatching { context.unbindService(connection) }
        }
    }

    @Test
    fun uncaughtJvmCrashInWorkerIsCapturedInCrashSummary() {
        val crash = crashWorkerAndClassify(DiffusionWorkerService.FAULT_JAVA_CRASH)
        assertTrue(
            "crashSummary should carry the uncaught-exception breadcrumb, got: ${crash.crashSummary}",
            crash.crashSummary?.contains("Injected uncaught JVM fault") == true,
        )
    }

    @Test
    fun nativeAbortInWorkerIsCapturedInCrashSummary() {
        val crash = crashWorkerAndClassify(DiffusionWorkerService.FAULT_NATIVE_ABORT)
        assertTrue(
            "crashSummary should carry the tombstone signal, got: ${crash.crashSummary} " +
                "(exitReason=${crash.exitReason})",
            crash.crashSummary?.contains("SIG") == true,
        )
    }

    /**
     * Binds the real worker process, injects [faultMode], triggers a generation so the fault
     * fires, waits for the process to die, then classifies the death exactly as the production
     * binderDied path does — polling until the crash detail (breadcrumb / tombstone trace)
     * becomes available so we also learn how long ApplicationExitInfo takes to expose it.
     */
    private fun crashWorkerAndClassify(faultMode: String): io.aatricks.llmedge.core.WorkerCrashedException {
        val connected = CountDownLatch(1)
        var binder: IBinder? = null
        val connection =
            object : ServiceConnection {
                override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                    binder = service
                    connected.countDown()
                }

                override fun onServiceDisconnected(name: ComponentName?) = Unit
            }
        try {
            assertTrue(
                context.bindService(
                    Intent(context, DiffusionWorkerService::class.java),
                    connection,
                    Context.BIND_AUTO_CREATE,
                ),
            )
            assertTrue("service did not connect", connected.await(20, TimeUnit.SECONDS))
            val worker = IDiffusionWorker.Stub.asInterface(binder)
            val workerPid = worker.pid
            val died = CountDownLatch(1)
            binder!!.linkToDeath({ died.countDown() }, 0)
            worker.installFaultInjection(
                Bundle().apply { putString(DiffusionWorkerService.FAULT_MODE, faultMode) },
            )
            val silentCallback =
                object : IDiffusionResultCallback.Stub() {
                    override fun onPhase(update: PhaseUpdate) = Unit

                    override fun onCompleted(result: IpcImageResult) = Unit

                    override fun onFailed(failure: IpcFailure) = Unit
                }
            worker.generateImage(
                IpcCodecs.toIpc(io.aatricks.llmedge.image.ImageGenerationRequest(prompt = "fault")),
                silentCallback,
            )
            assertTrue("worker process did not die", died.await(30, TimeUnit.SECONDS))

            // The tombstone trace is attached asynchronously after debuggerd finishes; poll so we
            // measure the latency instead of guessing. The breadcrumb is destructive to read, so
            // only classifications that still lack detail are retried.
            val start = SystemClock.uptimeMillis()
            var last: io.aatricks.llmedge.core.WorkerCrashedException? = null
            while (SystemClock.uptimeMillis() - start < 20_000) {
                val classified =
                    WorkerFailureClassifier.classify(
                        context = context,
                        pid = workerPid,
                        lastPhase = DiffusionPhases.REQUESTED,
                        lastBackend = "CPU",
                        killedByWatchdog = false,
                        stallMs = 0,
                    )
                if (classified is io.aatricks.llmedge.core.WorkerCrashedException) {
                    last = classified
                    val summary = classified.crashSummary ?: ""
                    if (summary.contains("Injected uncaught JVM fault") || summary.contains("SIG")) {
                        android.util.Log.i(
                            "IsolatedWorkerTest",
                            "crash detail available after ${SystemClock.uptimeMillis() - start} ms: $summary",
                        )
                        return classified
                    }
                }
                Thread.sleep(500)
            }
            assertTrue("worker death never classified as a crash", last != null)
            return last!!
        } finally {
            runCatching { context.unbindService(connection) }
        }
    }

    @Test
    fun watchdogKillsHungWorkerAndPersistsVerdict() {
        val config =
            LLMEdgeConfig(
                image =
                    ImageRuntimeConfig(
                        workerMode = DiffusionWorkerMode.ISOLATED_PROCESS,
                        hangRecoveryPolicy = HangRecoveryPolicy.FAIL_FAST,
                        watchdog =
                            WorkerWatchdogConfig(
                                cpuSampleIntervalMs = 500,
                                loadingStallTimeoutMs = 4_000,
                                generatingStallTimeoutMs = 4_000,
                                stepStallTimeoutMs = 4_000,
                                hardWallTimeoutMs = 120_000,
                            ),
                    ),
            )
        val engine = IsolatedDiffusionEngine(context, edgeScope, config)
        try {
            runBlocking {
                withTimeout(60_000) {
                    engine.installFaultInjectionForTests(
                        Bundle().apply {
                            putString(DiffusionWorkerService.FAULT_MODE, DiffusionWorkerService.FAULT_HANG_AFTER_PHASE)
                            putString(DiffusionWorkerService.FAULT_BACKEND, "VULKAN")
                        },
                    )
                    try {
                        engine.generate(ImageGenerationRequest(prompt = "hang test"))
                        fail("expected GenerationHangException")
                    } catch (hang: GenerationHangException) {
                        assertEquals("VULKAN", hang.backend)
                        assertEquals(DiffusionPhases.LOADING, hang.phase)
                    }
                }
            }
            assertTrue(
                "hang verdict must be persisted",
                BackendVerdictStore(context).load().contains(ComputeSubsystem.IMAGE to ComputeBackend.VULKAN),
            )
        } finally {
            engine.close()
        }
    }
}
