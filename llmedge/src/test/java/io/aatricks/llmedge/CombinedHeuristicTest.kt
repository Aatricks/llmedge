package io.aatricks.llmedge

import android.app.ActivityManager
import android.content.Context
import androidx.test.core.app.ApplicationProvider
// Shadows import will be used later
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import org.junit.After
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.Shadows
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import io.aatricks.llmedge.image.diffusion.StableDiffusion

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class CombinedHeuristicTest {
    @After
    fun tearDown() {
        try { unmockkAll() } catch (_: Throwable) {}
    }

    private fun callCompanionPrivateMethod(
        methodName: String,
        vararg args: Any?
    ): Any? {
        // computeEffectiveSequentialLoad signature: (Context, String, Boolean?, Boolean, ActivityManager?)
        val companion = StableDiffusion::class.java.getDeclaredField("Companion").get(null)
        val method = StableDiffusion::class.java.declaredMethods.find {
            it.name == methodName && it.parameterCount == args.size && it.parameterTypes[0] == Context::class.java
        } ?: throw NoSuchMethodException(methodName)
        method.isAccessible = true
        return method.invoke(companion, *args)
    }

    @Test
    fun `explicit sequential load true returns true`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val result = callCompanionPrivateMethod("computeEffectiveSequentialLoad", context, "modelPath", true, false, null) as Pair<Boolean, Long>
        assertTrue(result.first)
    }

    @Test
    fun `explicit sequential load false returns false`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val result = callCompanionPrivateMethod("computeEffectiveSequentialLoad", context, "modelPath", false, false, null) as Pair<Boolean, Long>
        assertFalse(result.first)
    }

    @Test
    fun `total system RAM under threshold triggers sequential load`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val shadowActivityManager = Shadows.shadowOf(activityManager)
        val mi = ActivityManager.MemoryInfo()
        mi.totalMem = 6L * 1024L * 1024L * 1024L
        mi.availMem = 2L * 1024L * 1024L * 1024L
        shadowActivityManager.setMemoryInfo(mi)

        // Mock estimateModelParamsMemoryBytes to a small value to ensure low RAM is decisive
        io.mockk.mockkObject(StableDiffusion.Companion)
        every { StableDiffusion.estimateModelParamsMemoryBytes(any(), any()) } returns 1L * 1024L * 1024L

        val result = callCompanionPrivateMethod("computeEffectiveSequentialLoad", context, "modelPath", null, false, activityManager) as Pair<Boolean, Long>
        assertTrue(result.first)
    }

    @org.junit.Ignore("Flaky in CI due to Runtime mocking")
    @Test
    fun `heap shortage triggers sequential load and preferPerformanceMode relaxes threshold`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val shadowActivityManager = Shadows.shadowOf(activityManager)
        val mi = ActivityManager.MemoryInfo()
        mi.totalMem = 16L * 1024L * 1024L * 1024L
        mi.availMem = 8L * 1024L * 1024L * 1024L
        shadowActivityManager.setMemoryInfo(mi)

        // Mock heap so it's small (heapAvail ~100MB)
        mockkStatic(Runtime::class)
        every { Runtime.getRuntime().totalMemory() } returns 1100L * 1024L * 1024L
        every { Runtime.getRuntime().freeMemory() } returns 1000L * 1024L * 1024L
        every { Runtime.getRuntime().maxMemory() } returns 1200L * 1024L * 1024L

        io.mockk.mockkObject(StableDiffusion.Companion)
        val estimateBytes = 80L * 1024L * 1024L
        every { StableDiffusion.estimateModelParamsMemoryBytes(any(), any()) } returns estimateBytes

        val resultDefault = callCompanionPrivateMethod("computeEffectiveSequentialLoad", context, "modelPath", null, false, activityManager) as Pair<Boolean, Long>
        assertTrue(resultDefault.first)

        val resultRelaxed = callCompanionPrivateMethod("computeEffectiveSequentialLoad", context, "modelPath", null, true, activityManager) as Pair<Boolean, Long>
        assertFalse(resultRelaxed.first)
    }
}
