package io.aatricks.llmedge.lifecycle

import android.os.Looper
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import io.aatricks.llmedge.LLMEdge
import io.mockk.mockk
import io.mockk.verify
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class LLMEdgeLifecycleTest {
    @Test
    fun bindMarshalsObserverRegistrationToMainThread() {
        val owner = TestLifecycleOwner()
        owner.registry.currentState = Lifecycle.State.CREATED
        val edge = mockk<LLMEdge>(relaxed = true)
        val result = AtomicReference<LLMEdge?>(null)
        val failure = AtomicReference<Throwable?>(null)

        val worker =
            Thread {
                try {
                    result.set(LLMEdgeLifecycle.bind(owner, edge))
                } catch (t: Throwable) {
                    failure.set(t)
                }
            }

        worker.start()
        val deadline = System.currentTimeMillis() + 2_000
        while (worker.isAlive && System.currentTimeMillis() < deadline) {
            shadowOf(Looper.getMainLooper()).idle()
            worker.join(25)
        }

        assertFalse("Background bind thread should have completed", worker.isAlive)
        assertNull(failure.get())
        assertSame(edge, result.get())

        owner.registry.currentState = Lifecycle.State.DESTROYED
        shadowOf(Looper.getMainLooper()).idle()

        verify(exactly = 1) { edge.close() }
    }

    private class TestLifecycleOwner : LifecycleOwner {
        val registry = LifecycleRegistry(this)

        override val lifecycle: Lifecycle
            get() = registry
    }
}