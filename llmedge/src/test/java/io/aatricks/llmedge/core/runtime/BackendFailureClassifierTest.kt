package io.aatricks.llmedge.core.runtime

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BackendFailureClassifierTest {
    @Test
    fun `recognizes backend failures from direct message`() {
        assertTrue(BackendFailureClassifier.isBackendFailure(IllegalStateException("backend device lost")))
    }

    @Test
    fun `recognizes backend failures from cause`() {
        val error = IllegalStateException("wrapper", IllegalStateException("backend unavailable"))
        assertTrue(BackendFailureClassifier.isBackendFailure(error))
    }

    @Test
    fun `ignores unrelated failures`() {
        assertFalse(BackendFailureClassifier.isBackendFailure(IllegalStateException("something else")))
    }
}
