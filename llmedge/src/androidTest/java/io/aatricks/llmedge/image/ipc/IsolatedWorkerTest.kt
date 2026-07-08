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
