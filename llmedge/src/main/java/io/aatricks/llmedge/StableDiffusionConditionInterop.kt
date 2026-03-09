package io.aatricks.llmedge

internal object StableDiffusionConditionInterop {
    fun fromNativeRaw(raw: Array<Any?>): PrecomputedCondition? {
        val cross = raw.getOrNull(0) as? FloatArray
        val crossDims = raw.getOrNull(1) as? IntArray
        val vector = raw.getOrNull(2) as? FloatArray
        val vectorDims = raw.getOrNull(3) as? IntArray
        val concat = raw.getOrNull(4) as? FloatArray
        val concatDims = raw.getOrNull(5) as? IntArray
        if (cross == null && vector == null && concat == null) {
            return null
        }
        return PrecomputedCondition(
            cCrossAttn = cross,
            cCrossAttnDims = crossDims,
            cVector = vector,
            cVectorDims = vectorDims,
            cConcat = concat,
            cConcatDims = concatDims,
        )
    }

    fun toNativeArray(condition: PrecomputedCondition?): Array<Any?>? =
        condition?.let {
            arrayOf<Any?>(
                it.cCrossAttn,
                it.cCrossAttnDims,
                it.cVector,
                it.cVectorDims,
                it.cConcat,
                it.cConcatDims,
            )
        }
}