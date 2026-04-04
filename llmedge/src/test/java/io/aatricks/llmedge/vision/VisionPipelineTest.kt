package io.aatricks.llmedge.vision

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.TextRuntimeConfig
import io.aatricks.llmedge.VisionRuntimeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.runtime.SmolLM
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import io.mockk.verify
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VisionPipelineTest {

    @Test
    fun `prepare caches runtime and analyze reuses it`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val resolver = mockk<ModelRepository>()
        val config =
            LLMEdgeConfig(
                text = TextRuntimeConfig(useVulkan = false, promptThreads = 4, generationThreads = 2),
                vision = VisionRuntimeConfig(useVulkan = false, promptThreads = 4, generationThreads = 2),
            )
        val smol = mockk<SmolLM>(relaxed = true)
        val projector = mockk<Projector>(relaxed = true)
        val model = mockk<ModelSpec>()
        val projectorSpec = mockk<ModelSpec>()
        val modelFile = File.createTempFile("llava-vision-model", ".gguf").apply { writeText("model") }
        val projectorFile = File.createTempFile("llava-vision-projector", ".mmproj.gguf").apply { writeText("projector") }
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        val edgeScope = LLMEdgeScope(this, config.text.promptThreads)

        coEvery { resolver.resolve(context, model) } returns modelFile
        coEvery { resolver.resolve(context, projectorSpec) } returns projectorFile
        every { model.cacheKey } returns "vision-model"
        every { projectorSpec.cacheKey } returns "vision-projector"
        coEvery { smol.load(any(), any(), any()) } returns Unit
        every { smol.getNativeModelPointer() } returns 33L
        every { smol.getActiveBackend() } returns io.aatricks.llmedge.runtime.ComputeBackend.CPU
        every { projector.isReady() } returns true
        every { projector.nativeHandle() } returns 77L
        every { smol.primeImageBuffer(77L, any(), 1) } returns true
        every { smol.getResponse("Describe the image", -1, SmolLM.DEFAULT_BLOCKING_BATCH_SIZE) } returns "warm-response"
        every { smol.getEstimatedNativeMemoryBytes() } returns 1024L
        every { smol.getEstimatedStateMemoryBytes() } returns 64L

        val pipeline =
            VisionPipeline(
                context = context,
                scope = edgeScope,
                resolver = resolver,
                config = config,
                smolLmFactory = { smol },
                projectorFactory = { projector },
            )

        try {
            pipeline.prepare(model = model, projector = projectorSpec, numThreads = 4, generationThreads = 2)
            val result =
                pipeline.analyze(
                    VisionRequest(
                        image = bitmap,
                        prompt = "Describe the image",
                        model = model,
                        projector = projectorSpec,
                        numThreads = 4,
                        generationThreads = 2,
                    ),
                )

            assertEquals("warm-response", result.text)
            coVerify(exactly = 1) { smol.load(modelFile.absolutePath, any(), any()) }
            verify(exactly = 1) { projector.init(projectorFile.absolutePath, 33L) }
            verify(exactly = 1) { smol.clearKvCache() }
        } finally {
            pipeline.close()
            edgeScope.close()
            modelFile.delete()
            projectorFile.delete()
        }
    }

    @Test
    fun `prepare uses config-backed runtime defaults`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val resolver = mockk<ModelRepository>()
        val config =
            LLMEdgeConfig(
                text = TextRuntimeConfig(useVulkan = false, promptThreads = 7, generationThreads = 3, useFlashAttention = true),
                vision = VisionRuntimeConfig(useVulkan = true, promptThreads = 7, generationThreads = 3, useFlashAttention = false),
            )
        val smol = mockk<SmolLM>(relaxed = true)
        val projector = mockk<Projector>(relaxed = true)
        val model = mockk<ModelSpec>()
        val projectorSpec = mockk<ModelSpec>()
        val modelFile = File.createTempFile("llava-default-model", ".gguf").apply { writeText("model") }
        val projectorFile = File.createTempFile("llava-default-projector", ".mmproj.gguf").apply { writeText("projector") }
        val paramsSlot = slot<SmolLM.InferenceParams>()
        val edgeScope = LLMEdgeScope(this, config.text.promptThreads)

        coEvery { resolver.resolve(context, model) } returns modelFile
        coEvery { resolver.resolve(context, projectorSpec) } returns projectorFile
        every { model.cacheKey } returns "default-model"
        every { projectorSpec.cacheKey } returns "default-projector"
        coEvery { smol.load(modelFile.absolutePath, capture(paramsSlot), any()) } returns Unit
        every { smol.getNativeModelPointer() } returns 44L
        every { smol.getActiveBackend() } returns io.aatricks.llmedge.runtime.ComputeBackend.VULKAN
        every { projector.isReady() } returns true

        val pipeline =
            VisionPipeline(
                context = context,
                scope = edgeScope,
                resolver = resolver,
                config = config,
                smolLmFactory = { useVulkan ->
                    assertEquals(true, useVulkan)
                    smol
                },
                projectorFactory = { projector },
            )

        try {
            pipeline.prepare(
                model = model,
                projector = projectorSpec,
                numThreads = config.vision.promptThreads,
                generationThreads = config.vision.generationThreads,
            )

            assertEquals(7, paramsSlot.captured.numThreads)
            assertEquals(3, paramsSlot.captured.generationThreads)
            assertEquals(false, paramsSlot.captured.useFlashAttn)
        } finally {
            pipeline.close()
            edgeScope.close()
            modelFile.delete()
            projectorFile.delete()
        }
    }
}
