package io.aatricks.llmedge.lifecycle

import android.os.Handler
import android.os.Looper
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import io.aatricks.llmedge.LLMEdge
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference

class LLMEdgeLifecycle private constructor(private val edge: LLMEdge) : DefaultLifecycleObserver {
    override fun onDestroy(owner: LifecycleOwner) {
        owner.lifecycle.removeObserver(this)
        edge.close()
    }

    companion object {
        private val mainHandler: Handler by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            Handler(Looper.getMainLooper())
        }

        @JvmStatic
        fun bind(owner: LifecycleOwner, edge: LLMEdge): LLMEdge {
            val observer = LLMEdgeLifecycle(edge)
            if (Looper.myLooper() == Looper.getMainLooper()) {
                owner.lifecycle.addObserver(observer)
                return edge
            }

            val failure = AtomicReference<Throwable?>(null)
            val latch = CountDownLatch(1)
            val posted =
                mainHandler.post {
                    try {
                        owner.lifecycle.addObserver(observer)
                    } catch (t: Throwable) {
                        failure.set(t)
                    } finally {
                        latch.countDown()
                    }
                }

            if (!posted) {
                throw IllegalStateException("Unable to bind LLMEdge lifecycle observer on the main thread")
            }

            try {
                latch.await()
            } catch (interrupted: InterruptedException) {
                Thread.currentThread().interrupt()
                throw IllegalStateException(
                    "Interrupted while binding LLMEdge to the owner lifecycle",
                    interrupted,
                )
            }

            failure.get()?.let { throw it }
            return edge
        }
    }
}
