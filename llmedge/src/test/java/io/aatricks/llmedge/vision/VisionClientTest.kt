package io.aatricks.llmedge.vision

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.TextRuntimeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.mockk.coVerify
import io.mockk.coEvery
import io.mockk.mockk
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VisionClientTest {

    @Test
    fun `concurrent analyze calls remain isolated and last runtime memory follows last completion`() = runTest {
        val context = mockk<Context>(relaxed = true)
        val pipeline = mockk<VisionPipeline>()
        val scope = mockk<LLMEdgeScope>(relaxed = true)
        val modelRepository = mockk<ModelRepository>()
        val bitmap = mockk<Bitmap>()
        val model = mockk<ModelSpec>()
        val projector = mockk<ModelSpec>()

        coEvery { pipeline.analyze(match { it.prompt == "first" }, any()) } coAnswers {
            delay(40)
            VisionPipelineResult(
                text = "first-result",
                runtimeMemory = VisionRuntimeMemory(nativeBytes = 100L, stateBytes = 10L),
            )
        }
        coEvery { pipeline.analyze(match { it.prompt == "second" }, any()) } coAnswers {
            delay(5)
            VisionPipelineResult(
                text = "second-result",
                runtimeMemory = VisionRuntimeMemory(nativeBytes = 200L, stateBytes = 20L),
            )
        }

        val client = VisionClient.forTesting(context, scope, LLMEdgeConfig(), modelRepository, pipeline)

        val first = async {
            client.analyze(VisionRequest(bitmap, "first", model, projector))
        }
        val second = async {
            client.analyze(VisionRequest(bitmap, "second", model, projector))
        }

        assertEquals("first-result", first.await())
        assertEquals("second-result", second.await())

        val memory = client.getLastRuntimeMemory()
        assertNotNull(memory)
        assertEquals(100L, memory?.nativeBytes)
        assertEquals(10L, memory?.stateBytes)
    }

    @Test
    fun `prepare forwards default runtime settings to pipeline`() = runTest {
        val context = mockk<Context>(relaxed = true)
        val pipeline = mockk<VisionPipeline>(relaxed = true)
        val scope = mockk<LLMEdgeScope>(relaxed = true)
        val modelRepository = mockk<ModelRepository>()
        val model = mockk<ModelSpec>()
        val projector = mockk<ModelSpec>()
        val config = LLMEdgeConfig(text = TextRuntimeConfig(promptThreads = 6, generationThreads = 3))
        val client = VisionClient.forTesting(context, scope, config, modelRepository, pipeline)

        client.prepare(model = model, projector = projector)

        coVerify(exactly = 1) {
            pipeline.prepare(
                model = model,
                projector = projector,
                numThreads = 6,
                generationThreads = 3,
                onStatus = null,
            )
        }
    }
}
