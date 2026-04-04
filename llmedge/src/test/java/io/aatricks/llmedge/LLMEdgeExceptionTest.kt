package io.aatricks.llmedge

import io.aatricks.llmedge.core.InferenceFailedException
import io.aatricks.llmedge.core.LLMEdgeException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LLMEdgeExceptionTest {
    @Test
    fun `inference failures remain llmedge exceptions`() {
        val error = InferenceFailedException(operation = "Video generation", detail = "")

        assertTrue(error is LLMEdgeException)
        assertEquals("Video generation failed", error.message)
    }
}
