package io.aatricks.llmedge.core

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ClientBootstrapTest {
    @Test
    fun `createOwned closes bootstrap scope when build fails`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val parentScope = CoroutineScope(SupervisorJob())
        lateinit var capturedBootstrap: ClientBootstrapContext

        try {
            ClientBootstrap.createOwned(
                context = context,
                scope = parentScope,
                inferenceThreads = 2,
            ) { bootstrap ->
                capturedBootstrap = bootstrap
                throw IllegalStateException("boom")
            }
            fail("Expected build failure to be rethrown")
        } catch (error: IllegalStateException) {
            assertSame(context.applicationContext, capturedBootstrap.appContext)
            assertTrue(
                capturedBootstrap.edgeScope.coroutineScope.coroutineContext[Job]?.isCancelled == true,
            )
            assertTrue(error.suppressed.isEmpty())
        } finally {
            parentScope.coroutineContext[Job]?.cancel()
        }
    }
}
