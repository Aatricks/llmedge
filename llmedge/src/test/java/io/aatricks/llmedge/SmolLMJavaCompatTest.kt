package io.aatricks.llmedge

import io.mockk.*
import kotlinx.coroutines.delay
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import io.aatricks.llmedge.text.runtime.SmolLM
import io.aatricks.llmedge.text.runtime.SmolLMJavaCompat

class SmolLMJavaCompatTest {

    @Test
    fun `InferenceParamsBuilder builds correct params`() {
        val builder = SmolLMJavaCompat.InferenceParamsBuilder()

        val params = builder
            .setMinP(0.05f)
            .setTemperature(0.5f)
            .setStoreChats(false)
            .setContextSize(2048L)
            .setChatTemplate("test template")
            .setNumThreads(8)
            .setUseMmap(false)
            .setUseMlock(true)
            .setThinkingMode(SmolLM.ThinkingMode.DISABLED)
            .setReasoningBudget(100)
            .build()

        assertEquals(0.05f, params.minP)
        assertEquals(0.5f, params.temperature)
        assertEquals(false, params.storeChats)
        assertEquals(2048L, params.contextSize)
        assertEquals("test template", params.chatTemplate)
        assertEquals(8, params.numThreads)
        assertEquals(false, params.useMmap)
        assertEquals(true, params.useMlock)
        assertEquals(SmolLM.ThinkingMode.DISABLED, params.thinkingMode)
        assertEquals(100, params.reasoningBudget)
    }

    @Test
    fun `InferenceParamsBuilder default values work correctly`() {
        val builder = SmolLMJavaCompat.InferenceParamsBuilder()
        val params = builder.build()

        assertEquals(0.1f, params.minP)
        assertEquals(0.8f, params.temperature)
        assertEquals(true, params.storeChats)
        assertEquals(null, params.contextSize)
        assertEquals(null, params.chatTemplate)
        assertEquals(4, params.numThreads)
        assertEquals(true, params.useMmap)
        assertEquals(false, params.useMlock)
        assertEquals(SmolLM.ThinkingMode.DEFAULT, params.thinkingMode)
        assertEquals(null, params.reasoningBudget)
    }

    @Test
    fun `loadBlocking works with mocked SmolLM`() {
        val mockSmol = mockk<SmolLM>()
        coEvery { mockSmol.load(any(), any()) } coAnswers { Unit }

        SmolLMJavaCompat.loadBlocking(mockSmol, "/fake/path/model.gguf", SmolLM.InferenceParams())

        coVerify { mockSmol.load("/fake/path/model.gguf", any()) }
    }

    @Test
    fun `loadBlocking works with null params`() {
        val mockSmol = mockk<SmolLM>()
        coEvery { mockSmol.load(any()) } coAnswers { Unit }

        SmolLMJavaCompat.loadBlocking(mockSmol, "/fake/path/model.gguf", null)

        coVerify { mockSmol.load("/fake/path/model.gguf") }
    }

    @Test
    fun `loadAsync calls callback on success with mocked SmolLM`() {
        val mockSmol = mockk<SmolLM>()
        coEvery { mockSmol.load(any(), any()) } coAnswers { Unit }

        val latch = CountDownLatch(1)
        var successCalled = false
        var errorCalled = false

        val callback = object : SmolLMJavaCompat.LoadCallback {
            override fun onSuccess() {
                successCalled = true
                latch.countDown()
            }
            override fun onError(t: Throwable) {
                errorCalled = true
                latch.countDown()
            }
        }

        SmolLMJavaCompat.loadAsync(mockSmol, "/fake/path/model.gguf", SmolLM.InferenceParams(), callback)

        // Wait for async operation to complete
        assert(latch.await(5, TimeUnit.SECONDS))
        assert(successCalled)
        assert(!errorCalled)
        coVerify { mockSmol.load("/fake/path/model.gguf", any()) }
    }

    @Test
    fun `loadAsync calls callback on error with mocked SmolLM`() {
        val mockSmol = mockk<SmolLM>()
        val testException = RuntimeException("Test error")
        coEvery { mockSmol.load(any(), any()) } throws testException

        val latch = CountDownLatch(1)
        var successCalled = false
        var errorCalled = false
        var receivedThrowable: Throwable? = null

        val callback = object : SmolLMJavaCompat.LoadCallback {
            override fun onSuccess() {
                successCalled = true
                latch.countDown()
            }
            override fun onError(t: Throwable) {
                errorCalled = true
                receivedThrowable = t
                latch.countDown()
            }
        }

        SmolLMJavaCompat.loadAsync(mockSmol, "/fake/path/model.gguf", SmolLM.InferenceParams(), callback)

        // Wait for async operation to complete
        assert(latch.await(5, TimeUnit.SECONDS))
        assert(!successCalled)
        assert(errorCalled)
        assertEquals(testException, receivedThrowable)
    }

    @Test
    fun `streamResponse forwards chunks correctly with mocked SmolLM`() {
        val mockSmol = mockk<SmolLM>()
        val mockFlow = kotlinx.coroutines.flow.flow {
            emit("Hello")
            emit(" ")
            emit("world")
            emit("[EOG]")
        }
        coEvery { mockSmol.getResponseAsFlow(any()) } returns mockFlow

        val latch = CountDownLatch(1)
        val chunks = mutableListOf<String>()
        var completed = false
        var error: Throwable? = null

        val listener = object : SmolLMJavaCompat.StreamListener {
            override fun onChunk(chunk: String) {
                chunks.add(chunk)
            }
            override fun onComplete() {
                completed = true
                latch.countDown()
            }
            override fun onError(t: Throwable) {
                error = t
                latch.countDown()
            }
        }

        SmolLMJavaCompat.streamResponse(mockSmol, "Test query", listener)

        // Wait for async operation to complete
        assert(latch.await(5, TimeUnit.SECONDS))
        assert(completed)
        assert(error == null)
        assertEquals(listOf("Hello", " ", "world"), chunks)
        // Ensure we passed the exact query through to the SmolLM instance (strict verification)
        coVerify { mockSmol.getResponseAsFlow("Test query") }
    }

    @Test
    fun `shutdownCompatDispatcher works without throwing`() {
        // Should not throw any exceptions
        SmolLMJavaCompat.shutdownCompatDispatcher()
    }
}