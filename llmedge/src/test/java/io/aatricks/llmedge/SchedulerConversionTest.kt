package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SchedulerConversionTest {

    @Test
    fun `SampleMethod enum has correct native IDs`() {
        assertEquals(0, SampleMethod.DEFAULT.id)
        assertEquals(1, SampleMethod.EULER.id)
        assertEquals(2, SampleMethod.HEUN.id)
        assertEquals(3, SampleMethod.DPM2.id)
        assertEquals(4, SampleMethod.DPMPP2S_A.id)
        assertEquals(5, SampleMethod.DPMPP2M.id)
        assertEquals(6, SampleMethod.DPMPP2MV2.id)
        assertEquals(7, SampleMethod.IPNDM.id)
        assertEquals(8, SampleMethod.IPNDM_V.id)
        assertEquals(9, SampleMethod.LCM.id)
        assertEquals(10, SampleMethod.DDIM_TRAILING.id)
        assertEquals(11, SampleMethod.TCD.id)
        assertEquals(12, SampleMethod.EULER_A.id)
    }

    @Test
    fun `Scheduler enum has correct native IDs`() {
        assertEquals(0, Scheduler.DEFAULT.id)
        assertEquals(1, Scheduler.DISCRETE.id)
        assertEquals(2, Scheduler.KARRAS.id)
        assertEquals(3, Scheduler.EXPONENTIAL.id)
        assertEquals(4, Scheduler.AYS.id)
        assertEquals(5, Scheduler.GITS.id)
        assertEquals(6, Scheduler.SGM_UNIFORM.id)
        assertEquals(7, Scheduler.SIMPLE.id)
        assertEquals(8, Scheduler.SMOOTHSTEP.id)
    }

    @Test
    fun `SampleMethod fromId returns correct enum`() {
        assertEquals(SampleMethod.DEFAULT, SampleMethod.fromId(0))
        assertEquals(SampleMethod.EULER, SampleMethod.fromId(1))
        assertEquals(SampleMethod.EULER_A, SampleMethod.fromId(12))
        assertEquals(SampleMethod.DEFAULT, SampleMethod.fromId(999)) // Invalid ID
    }

    @Test
    fun `Scheduler fromId returns correct enum`() {
        assertEquals(Scheduler.DEFAULT, Scheduler.fromId(0))
        assertEquals(Scheduler.KARRAS, Scheduler.fromId(2))
        assertEquals(Scheduler.SMOOTHSTEP, Scheduler.fromId(8))
        assertEquals(Scheduler.DEFAULT, Scheduler.fromId(999)) // Invalid ID
    }

    @Test
    fun `all SampleMethod values have valid IDs`() {
        val sampleMethods = SampleMethod.values()
        sampleMethods.forEach { method ->
            assertTrue("SampleMethod $method should have non-negative ID", method.id >= 0)
            assertTrue("SampleMethod $method should have ID less than enum count", method.id < sampleMethods.size)
        }
    }

    @Test
    fun `all Scheduler values have valid IDs`() {
        val schedulers = Scheduler.values()
        schedulers.forEach { scheduler ->
            assertTrue("Scheduler $scheduler should have non-negative ID", scheduler.id >= 0)
            assertTrue("Scheduler $scheduler should have ID less than enum count", scheduler.id < schedulers.size)
        }
    }

    @Test
    fun `SampleMethod enum has all expected values`() {
        val expectedMethods = arrayOf(
            SampleMethod.DEFAULT,
            SampleMethod.EULER,
            SampleMethod.HEUN,
            SampleMethod.DPM2,
            SampleMethod.DPMPP2S_A,
            SampleMethod.DPMPP2M,
            SampleMethod.DPMPP2MV2,
            SampleMethod.IPNDM,
            SampleMethod.IPNDM_V,
            SampleMethod.LCM,
            SampleMethod.DDIM_TRAILING,
            SampleMethod.TCD,
            SampleMethod.EULER_A
        )

        val actualMethods = SampleMethod.values()
        assertEquals(expectedMethods.size, actualMethods.size)

        expectedMethods.forEach { expected ->
            assertTrue("SampleMethod $expected should exist", actualMethods.contains(expected))
        }
    }

    @Test
    fun `Scheduler enum has all expected values`() {
        val expectedSchedulers = arrayOf(
            Scheduler.DEFAULT,
            Scheduler.DISCRETE,
            Scheduler.KARRAS,
            Scheduler.EXPONENTIAL,
            Scheduler.AYS,
            Scheduler.GITS,
            Scheduler.SGM_UNIFORM,
            Scheduler.SIMPLE,
            Scheduler.SMOOTHSTEP
        )

        val actualSchedulers = Scheduler.values()
        assertEquals(expectedSchedulers.size, actualSchedulers.size)

        expectedSchedulers.forEach { expected ->
            assertTrue("Scheduler $expected should exist", actualSchedulers.contains(expected))
        }
    }
}