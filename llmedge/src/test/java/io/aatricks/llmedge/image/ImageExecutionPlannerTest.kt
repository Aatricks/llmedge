package io.aatricks.llmedge.image

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.ProgressEvent
import io.aatricks.llmedge.image.ipc.DiffusionPhases
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import java.io.File
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ImageExecutionPlannerTest {
    private val gib = 1024L * 1024L * 1024L

    @Test
    fun `image requests default to automatic sequential planning`() {
        assertNull(ImageGenerationRequest(prompt = "x").sequential)
    }

    @Test
    fun `automatic planner keeps a fitting masked T5 request direct`() {
        val decision =
            ImageExecutionPlanner.decide(
                sequential = null,
                recipe = maskedT5Recipe(),
                componentSizes = mapOf(
                    ImageExecutionComponentKind.TEXT_ENCODER to gib,
                    ImageExecutionComponentKind.DIFFUSION_MODEL to gib,
                ),
                memory = ImageMemorySnapshot(
                    availableSystemBytes = 7 * gib,
                    lowMemoryThresholdBytes = gib / 2,
                    totalSystemBytes = 8 * gib,
                ),
            )

        assertEquals(ImageExecutionMode.DIRECT, decision.mode)
        assertEquals("AUTO_DIRECT_FITS", decision.reason)
    }

    @Test
    fun `automatic planner stages a masked T5 request when combined weights exceed budget`() {
        val decision =
            ImageExecutionPlanner.decide(
                sequential = null,
                recipe = maskedT5Recipe(),
                componentSizes = mapOf(
                    ImageExecutionComponentKind.TEXT_ENCODER to (2 * gib),
                    ImageExecutionComponentKind.DIFFUSION_MODEL to (4 * gib),
                ),
                memory = ImageMemorySnapshot(
                    availableSystemBytes = 5 * gib,
                    lowMemoryThresholdBytes = gib / 2,
                    totalSystemBytes = 8 * gib,
                ),
            )

        assertEquals(ImageExecutionMode.SEQUENTIAL, decision.mode)
        assertEquals("AUTO_SEQUENTIAL_MEMORY", decision.reason)
    }

    @Test
    fun `manual overrides take precedence over automatic estimate`() {
        val recipe = maskedT5Recipe()
        val sizes =
            mapOf(
                ImageExecutionComponentKind.TEXT_ENCODER to (2 * gib),
                ImageExecutionComponentKind.DIFFUSION_MODEL to (4 * gib),
            )
        val memory =
            ImageMemorySnapshot(
                availableSystemBytes = 5 * gib,
                lowMemoryThresholdBytes = gib / 2,
                totalSystemBytes = 8 * gib,
            )

        assertEquals(
            ImageExecutionMode.DIRECT,
            ImageExecutionPlanner.decide(false, recipe, sizes, memory).mode,
        )
        assertEquals(
            ImageExecutionMode.SEQUENTIAL,
            ImageExecutionPlanner.decide(true, recipe, sizes, memory).mode,
        )
    }

    @Test
    fun `forced staged execution rejects a non splittable request`() {
        val error =
            runCatching {
                ImageExecutionPlanner.decide(
                    sequential = true,
                    recipe = ImageExecutionRecipe.none(),
                    componentSizes = emptyMap(),
                    memory = ImageMemorySnapshot(gib, 0, gib),
                )
            }.exceptionOrNull()

        assertTrue(error is UnsupportedStagedExecutionException)
    }

    @Test
    fun `recipe inference recognizes masked T5 topology without a model name`() {
        val request =
            ImageGenerationRequest(
                prompt = "x",
                model = io.aatricks.llmedge.model.ModelSpec.LocalFile(File("dit.safetensors")),
                textEncoder = io.aatricks.llmedge.model.ModelSpec.LocalFile(File("flan-t5-large.safetensors")),
                diffusionModelOnly = true,
            )

        assertEquals(
            ImageConditioningProfile.MASKED_T5,
            ImageExecutionPlanner.recipeFor(request).profile,
        )
    }

    @Test
    fun `default selector resolves every masked T5 component before applying the automatic estimate`() = runTest {
        val dit = ModelSpec.LocalFile(File("dit.safetensors"))
        val encoder = ModelSpec.LocalFile(File("flan-t5-large.safetensors"))
        val resolved = File.createTempFile("planner", ".bin").apply { deleteOnExit() }
        val requested = mutableListOf<ModelSpec>()
        val phases = mutableListOf<String>()
        val selector =
            DefaultImageExecutionPlanSelector(
                context = ApplicationProvider.getApplicationContext<Context>(),
                resolver =
                    object : ModelRepository {
                        override suspend fun resolve(
                            context: Context,
                            spec: ModelSpec,
                            onProgress: ((ProgressEvent.Downloading) -> Unit)?,
                        ): File {
                            assertEquals(listOf(DiffusionPhases.RESOLVING_MODEL), phases)
                            requested += spec
                            return resolved
                        }
                    },
                memorySnapshotProvider = {
                    ImageMemorySnapshot(
                        availableSystemBytes = 5 * gib,
                        lowMemoryThresholdBytes = gib / 2,
                        totalSystemBytes = 8 * gib,
                    )
                },
                fileSizeProvider = { spec, _ -> if (spec === dit) 4 * gib else 2 * gib },
                phaseListener =
                    object : DiffusionPhaseListener {
                        override fun onPhase(phase: String, backend: String?) {
                            phases += phase
                        }

                        override fun onStep(step: Int, totalSteps: Int) = Unit
                    },
            )

        val decision =
            selector.decide(
                ImageGenerationRequest(
                    prompt = "x",
                    model = dit,
                    textEncoder = encoder,
                    diffusionModelOnly = true,
                ),
                LLMEdgeConfig(),
            )

        assertEquals(ImageExecutionMode.SEQUENTIAL, decision.mode)
        assertEquals(listOf(encoder, dit), requested)
        assertEquals(listOf(DiffusionPhases.RESOLVING_MODEL), phases)
    }

    private fun maskedT5Recipe(): ImageExecutionRecipe =
        ImageExecutionRecipe(
            profile = ImageConditioningProfile.MASKED_T5,
            phases =
                listOf(
                    listOf(ImageExecutionComponentKind.TEXT_ENCODER),
                    listOf(ImageExecutionComponentKind.DIFFUSION_MODEL),
                ),
        )
}
