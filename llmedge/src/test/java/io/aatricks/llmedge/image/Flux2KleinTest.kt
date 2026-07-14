package io.aatricks.llmedge.image

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class Flux2KleinTest {
    @Test
    fun `standard request leaves execution mode automatic`() {
        val request = Flux2Klein.imageRequest(prompt = "a fox")

        assertEquals(Flux2Klein.diffusionModel, request.model)
        assertNull(request.sequential)
    }
}
