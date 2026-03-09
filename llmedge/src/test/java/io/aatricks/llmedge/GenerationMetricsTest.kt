package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import io.aatricks.llmedge.image.diffusion.GenerationMetrics

class GenerationMetricsTest {

    @Test
    fun `GenerationMetrics computed properties work correctly`() {
        val metrics = GenerationMetrics(
            totalTimeSeconds = 10.5f,
            framesPerSecond = 2.5f,
            timePerStep = 0.5f,
            peakMemoryUsageMb = 512L,
            vulkanEnabled = true,
            frameConversionTimeSeconds = 1.2f
        )

        assertEquals(0.4f, metrics.averageFrameTime) // 1/2.5
        assertEquals(2.0f, metrics.stepsPerSecond) // 1/0.5
        assertEquals("2.50 fps", metrics.throughput)
    }

    @Test
    fun `GenerationMetrics handles zero values gracefully`() {
        val metrics = GenerationMetrics(
            totalTimeSeconds = 0f,
            framesPerSecond = 0f,
            timePerStep = 0f,
            peakMemoryUsageMb = 0L,
            vulkanEnabled = false,
            frameConversionTimeSeconds = 0f
        )

        assertEquals(0f, metrics.averageFrameTime)
        assertEquals(0f, metrics.stepsPerSecond)
        assertEquals("0.00 fps", metrics.throughput)
    }

    @Test
    fun `GenerationMetrics toPrettyString formats output correctly`() {
        val metrics = GenerationMetrics(
            totalTimeSeconds = 10.5f,
            framesPerSecond = 2.5f,
            timePerStep = 0.5f,
            peakMemoryUsageMb = 512L,
            vulkanEnabled = true,
            frameConversionTimeSeconds = 1.2f
        )

        val prettyString = metrics.toPrettyString()
        assertTrue(prettyString.contains("Total time: 10.50s"))
        assertTrue(prettyString.contains("Throughput: 2.50 fps"))
        assertTrue(prettyString.contains("Average time/step: 0.500s"))
        assertTrue(prettyString.contains("Peak memory: 512MB"))
        assertTrue(prettyString.contains("Vulkan: enabled"))
        assertTrue(prettyString.contains("Frame conversion: 1.20s"))
    }

    @Test
    fun `GenerationMetrics toPrettyString handles disabled vulkan`() {
        val metrics = GenerationMetrics(
            totalTimeSeconds = 5.0f,
            framesPerSecond = 1.0f,
            timePerStep = 1.0f,
            peakMemoryUsageMb = 256L,
            vulkanEnabled = false,
            frameConversionTimeSeconds = 0.5f
        )

        val prettyString = metrics.toPrettyString()
        assertTrue(prettyString.contains("Vulkan: disabled"))
    }

    @Test
    fun `GenerationMetrics constructor accepts all parameters`() {
        val metrics = GenerationMetrics(
            totalTimeSeconds = 15.75f,
            framesPerSecond = 3.2f,
            timePerStep = 0.25f,
            peakMemoryUsageMb = 1024L,
            vulkanEnabled = true,
            frameConversionTimeSeconds = 2.1f
        )

        assertEquals(15.75f, metrics.totalTimeSeconds)
        assertEquals(3.2f, metrics.framesPerSecond)
        assertEquals(0.25f, metrics.timePerStep)
        assertEquals(1024L, metrics.peakMemoryUsageMb)
        assertEquals(true, metrics.vulkanEnabled)
        assertEquals(2.1f, metrics.frameConversionTimeSeconds)
    }

    @Test
    fun `GenerationMetrics handles very small values`() {
        val metrics = GenerationMetrics(
            totalTimeSeconds = 0.001f,
            framesPerSecond = 0.1f,
            timePerStep = 0.01f,
            peakMemoryUsageMb = 1L,
            vulkanEnabled = false,
            frameConversionTimeSeconds = 0.005f
        )

        assertEquals(10.0f, metrics.averageFrameTime) // 1/0.1
        assertEquals(100.0f, metrics.stepsPerSecond) // 1/0.01
        assertEquals("0.10 fps", metrics.throughput)
    }

    @Test
    fun `GenerationMetrics handles large values`() {
        val metrics = GenerationMetrics(
            totalTimeSeconds = 1000.0f,
            framesPerSecond = 60.0f,
            timePerStep = 0.1f,
            peakMemoryUsageMb = 8192L,
            vulkanEnabled = true,
            frameConversionTimeSeconds = 50.0f
        )

        assertEquals(0.016666668f, metrics.averageFrameTime, 0.0001f) // 1/60
        assertEquals(10.0f, metrics.stepsPerSecond) // 1/0.1
        assertEquals("60.00 fps", metrics.throughput)
    }
}