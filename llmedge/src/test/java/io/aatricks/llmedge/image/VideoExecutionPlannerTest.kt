package io.aatricks.llmedge.image

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.InsufficientMemoryException
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.image.ipc.DiffusionPhases
import io.aatricks.llmedge.model.ModelRegistry
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.model.VideoModels
import io.mockk.Called
import io.mockk.mockk
import io.mockk.verify
import java.io.File
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoExecutionPlannerTest {
    private val gib = 1024L * 1024L * 1024L

    @Test
    fun `forced sequential remains sequential when it fits`() {
        val decision =
            VideoExecutionPlanner.decide(
                forceSequentialLoad = true,
                sizes = VideoComponentSizes(diffusionBytes = gib, vaeBytes = gib / 4, textEncoderBytes = gib),
                memory = VideoMemorySnapshot(availableSystemBytes = 8 * gib, lowMemoryThresholdBytes = gib / 2, totalSystemBytes = 12 * gib),
            )

        assertEquals(ImageExecutionMode.SEQUENTIAL, decision.mode)
        assertEquals("FORCED_SEQUENTIAL", decision.reason)
    }

    @Test
    fun `automatic planner keeps video direct with two times headroom`() {
        val decision =
            VideoExecutionPlanner.decide(
                forceSequentialLoad = false,
                sizes = VideoComponentSizes(diffusionBytes = gib, vaeBytes = gib / 4, textEncoderBytes = gib),
                memory = VideoMemorySnapshot(availableSystemBytes = 8 * gib, lowMemoryThresholdBytes = gib / 2, totalSystemBytes = 12 * gib),
            )

        assertEquals(ImageExecutionMode.DIRECT, decision.mode)
        assertEquals("AUTO_DIRECT_HEADROOM", decision.reason)
    }

    @Test
    fun `automatic planner stages video when direct lacks two times headroom`() {
        val decision =
            VideoExecutionPlanner.decide(
                forceSequentialLoad = false,
                sizes = VideoComponentSizes(diffusionBytes = 2 * gib, vaeBytes = gib / 2, textEncoderBytes = 2 * gib),
                memory = VideoMemorySnapshot(availableSystemBytes = 8 * gib, lowMemoryThresholdBytes = gib / 2, totalSystemBytes = 12 * gib),
            )

        assertEquals(ImageExecutionMode.SEQUENTIAL, decision.mode)
        assertEquals("AUTO_SEQUENTIAL_MEMORY", decision.reason)
    }

    @Test
    fun `automatic planner refuses before loading when sequential peak exceeds budget`() {
        val error =
            runCatching {
                VideoExecutionPlanner.decide(
                    forceSequentialLoad = false,
                    sizes = VideoComponentSizes(diffusionBytes = 3 * gib, vaeBytes = gib, textEncoderBytes = 3 * gib),
                    memory = VideoMemorySnapshot(availableSystemBytes = 4 * gib, lowMemoryThresholdBytes = gib / 2, totalSystemBytes = 8 * gib),
                )
            }.exceptionOrNull()

        assertTrue(error is InsufficientMemoryException)
        assertTrue(error?.message.orEmpty().contains("sequential video loading"))
    }

    @Test
    fun `forced sequential planner refuses when sequential peak exceeds budget`() {
        val error =
            runCatching {
                VideoExecutionPlanner.decide(
                    forceSequentialLoad = true,
                    sizes = VideoComponentSizes(diffusionBytes = 3 * gib, vaeBytes = gib, textEncoderBytes = 3 * gib),
                    memory = VideoMemorySnapshot(availableSystemBytes = 4 * gib, lowMemoryThresholdBytes = gib / 2, totalSystemBytes = 8 * gib),
                )
            }.exceptionOrNull()

        assertTrue(error is InsufficientMemoryException)
    }

    @Test
    fun `planner refuses S22 Wan Q3 load with only boundary headroom`() {
        val error =
            runCatching {
                VideoExecutionPlanner.decide(
                    forceSequentialLoad = false,
                    sizes =
                        VideoComponentSizes(
                            diffusionBytes = 654_903_520,
                            vaeBytes = 253_815_318,
                            textEncoderBytes = 2_858_489_696,
                        ),
                    memory =
                        VideoMemorySnapshot(
                            availableSystemBytes = 4_373_471_232,
                            lowMemoryThresholdBytes = 408_944_640,
                            totalSystemBytes = 7_624_486_912,
                        ),
                )
            }.exceptionOrNull()

        assertTrue(error is InsufficientMemoryException)
    }

    @Test
    fun `executor planning refusal prevents runtime acquisition`() = runTest {
        val scope = LLMEdgeScope(this, 1)
        val requestExecutor = mockk<DiffusionRequestExecutor>(relaxed = true)
        val executor =
            VideoGenerationExecutor(
                scope = scope,
                config = LLMEdgeConfig(),
                generationMutex = Mutex(),
                state = ImageClientState(),
                requestExecutor = requestExecutor,
                executionPlanSelector =
                    VideoExecutionPlanSelector { _, _ ->
                        throw InsufficientMemoryException(
                            requiredBytes = 4 * gib,
                            availableBytes = 3 * gib,
                            operation = "sequential video loading",
                        )
                    },
            )

        try {
            val error = runCatching { executor.generate(VideoGenerationRequest(prompt = "x")).collect() }.exceptionOrNull()

            assertTrue(error is InsufficientMemoryException)
            verify { requestExecutor wasNot Called }
        } finally {
            scope.close()
        }
    }

    @Test
    fun `automatic planner preserves direct loading when memory telemetry is unavailable`() {
        val decision =
            VideoExecutionPlanner.decide(
                forceSequentialLoad = false,
                sizes = VideoComponentSizes(diffusionBytes = gib, vaeBytes = gib / 4, textEncoderBytes = gib),
                memory = VideoMemorySnapshot(availableSystemBytes = 0L, lowMemoryThresholdBytes = 0L, totalSystemBytes = 0L),
            )

        assertEquals(ImageExecutionMode.DIRECT, decision.mode)
        assertEquals("AUTO_DIRECT_MEMORY_UNAVAILABLE", decision.reason)
    }

    @Test
    fun `default selector resolves components and uses resolved file lengths`() = runTest {
        val model = ModelSpec.localFile(File("model.gguf"))
        val vae = ModelSpec.localFile(File("vae.safetensors"))
        val encoder = ModelSpec.localFile(File("encoder.gguf"))
        val files =
            mapOf(
                model to tempFileOfSize("model", 10),
                vae to tempFileOfSize("vae", 20),
                encoder to tempFileOfSize("encoder", 30),
            )
        val requested = mutableListOf<ModelSpec>()
        val phases = mutableListOf<String>()
        val config = LLMEdgeConfig(models = ModelRegistry(video = VideoModels(model, vae, encoder)))
        val selector =
            DefaultVideoExecutionPlanSelector(
                context = ApplicationProvider.getApplicationContext<Context>(),
                resolver =
                    object : ModelRepository {
                        override suspend fun resolve(
                            context: Context,
                            spec: ModelSpec,
                            onProgress: ((ProgressEvent.Downloading) -> Unit)?,
                        ): File {
                            requested += spec
                            return files.getValue(spec)
                        }
                    },
                memorySnapshotProvider = {
                    VideoMemorySnapshot(availableSystemBytes = 8 * gib, lowMemoryThresholdBytes = 0, totalSystemBytes = 8 * gib)
                },
                phaseListener =
                    object : DiffusionPhaseListener {
                        override fun onPhase(phase: String, backend: String?) {
                            phases += phase
                        }

                        override fun onStep(step: Int, totalSteps: Int) = Unit
                    },
            )

        val decision = selector.decide(VideoGenerationRequest(prompt = "x"), config)

        assertEquals(listOf(model, vae, encoder), requested)
        assertEquals(listOf(DiffusionPhases.RESOLVING_MODEL), phases)
        assertEquals(60L + 7L + 256L * 1024L * 1024L, decision.directPeakBytes)
    }

    private fun tempFileOfSize(prefix: String, size: Int): File =
        File.createTempFile(prefix, ".bin").apply {
            writeBytes(ByteArray(size))
            deleteOnExit()
        }
}
