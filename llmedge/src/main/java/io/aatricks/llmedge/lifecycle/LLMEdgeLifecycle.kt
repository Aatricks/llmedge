package io.aatricks.llmedge.lifecycle

import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import io.aatricks.llmedge.LLMEdge

class LLMEdgeLifecycle private constructor(private val edge: LLMEdge) : DefaultLifecycleObserver {
    override fun onDestroy(owner: LifecycleOwner) {
        owner.lifecycle.removeObserver(this)
        edge.close()
    }

    companion object {
        @JvmStatic
        fun bind(owner: LifecycleOwner, edge: LLMEdge): LLMEdge {
            owner.lifecycle.addObserver(LLMEdgeLifecycle(edge))
            return edge
        }
    }
}
