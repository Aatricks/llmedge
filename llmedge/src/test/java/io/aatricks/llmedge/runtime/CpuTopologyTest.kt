package io.aatricks.llmedge.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class CpuTopologyTest {

    @Test
    fun `homogeneous topology keeps all available processors with a floor of 2`() {
        val homogeneousCoreInfo = CpuTopology.CoreInfo(
            totalCores = 8,
            performanceCores = 8,
            efficiencyCores = 0,
            maxFrequencies = listOf(2000L, 2000L, 2000L, 2000L, 2000L, 2000L, 2000L, 2000L)
        )

        assertEquals(8, CpuTopology.selectDiffusionThreadCount(homogeneousCoreInfo, 8))
        assertEquals(4, CpuTopology.selectDiffusionThreadCount(homogeneousCoreInfo, 4))
        assertEquals(2, CpuTopology.selectDiffusionThreadCount(homogeneousCoreInfo, 2))
        assertEquals(2, CpuTopology.selectDiffusionThreadCount(homogeneousCoreInfo, 1))
    }

    @Test
    fun `heterogeneous topology with efficiency cores reserves exactly one available processor with a floor of 2`() {
        val heterogeneousCoreInfo = CpuTopology.CoreInfo(
            totalCores = 8,
            performanceCores = 4,
            efficiencyCores = 4,
            maxFrequencies = listOf(2800L, 2800L, 2800L, 2800L, 1800L, 1800L, 1800L, 1800L)
        )

        assertEquals(7, CpuTopology.selectDiffusionThreadCount(heterogeneousCoreInfo, 8))
        assertEquals(3, CpuTopology.selectDiffusionThreadCount(heterogeneousCoreInfo, 4))
        assertEquals(2, CpuTopology.selectDiffusionThreadCount(heterogeneousCoreInfo, 3))
        assertEquals(2, CpuTopology.selectDiffusionThreadCount(heterogeneousCoreInfo, 2))
        assertEquals(2, CpuTopology.selectDiffusionThreadCount(heterogeneousCoreInfo, 1))
    }
}
