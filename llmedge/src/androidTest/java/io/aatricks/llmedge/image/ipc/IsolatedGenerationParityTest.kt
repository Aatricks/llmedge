package io.aatricks.llmedge.image.ipc

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.DiffusionWorkerMode
import io.aatricks.llmedge.ImageRuntimeConfig
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Real-generation parity between the in-process and isolated engines, CPU backend (the path a
 * device with a broken Vulkan driver lands on). Opt-in: needs a local SD model, e.g.
 *   -Pandroid.testInstrumentationRunnerArguments.sdModelPath=/data/local/tmp/sdturbo.gguf
 */
@RunWith(AndroidJUnit4::class)
class IsolatedGenerationParityTest {
    @Test
    fun isolatedCpuGenerateMatchesInProcess() {
        val modelPath = InstrumentationRegistry.getArguments().getString("sdModelPath").orEmpty()
        assumeTrue(
            "Provide -Pandroid.testInstrumentationRunnerArguments.sdModelPath=<device path>",
            modelPath.isNotBlank() && File(modelPath).exists(),
        )
        val request =
            ImageGenerationRequest(
                prompt = "a red apple on a wooden table",
                width = 128,
                height = 128,
                steps = 2,
                cfgScale = 1.0f,
                seed = 7L,
                model = ModelSpec.localFile(modelPath),
            )

        val inProcess = generateWith(DiffusionWorkerMode.IN_PROCESS, request)
        val isolated = generateWith(DiffusionWorkerMode.ISOLATED_PROCESS, request)

        assertEquals(128, isolated.width)
        assertEquals(128, isolated.height)
        assertTrue("isolated output must not be uniform", isNotUniform(isolated))
        assertTrue("in-process output must not be uniform", isNotUniform(inProcess))

        // Same seed, same backend, same thread planning: outputs should match near-exactly.
        val total = 128 * 128
        var matching = 0
        for (y in 0 until 128) {
            for (x in 0 until 128) {
                if (inProcess.getPixel(x, y) == isolated.getPixel(x, y)) matching++
            }
        }
        assertTrue(
            "outputs diverged: only $matching/$total pixels identical",
            matching >= (total * 0.98).toInt(),
        )
    }

    @Test
    fun isolatedFlux2SequentialGenerate() {
        val ditPath = "/data/local/tmp/bonsai-flux2-klein-ternary-q2_k.gguf"
        val encoderPath = "/data/local/tmp/Qwen_3_4b-Q3_K_M.gguf"
        val vaePath = "/data/local/tmp/flux2-vae.safetensors"

        assumeTrue(
            "Model files missing on device",
            File(ditPath).exists() && File(encoderPath).exists() && File(vaePath).exists(),
        )

        val request =
            ImageGenerationRequest(
                prompt = "a red fox in snow",
                width = 256,
                height = 256,
                steps = 2,
                cfgScale = 1.0f,
                seed = 42L,
                model =
                    ModelSpec.localFile(
                        ditPath,
                        ModelHints(
                            artifactKind = ModelArtifactKind.DIFFUSION_MODEL,
                            capabilities = setOf(ModelCapability.IMAGE),
                        ),
                    ),
                vae =
                    ModelSpec.localFile(
                        vaePath,
                        ModelHints(
                            artifactKind = ModelArtifactKind.VAE,
                            capabilities = setOf(ModelCapability.IMAGE),
                        ),
                    ),
                textEncoder =
                    ModelSpec.localFile(
                        encoderPath,
                        ModelHints(
                            artifactKind = ModelArtifactKind.TEXT_ENCODER,
                            capabilities = setOf(ModelCapability.TEXT, ModelCapability.IMAGE),
                        ),
                    ),
                splitDiffusionModel = true,
                sequential = true,
            )

        val isolated = generateWith(DiffusionWorkerMode.ISOLATED_PROCESS, request)
        assertEquals(256, isolated.width)
        assertEquals(256, isolated.height)
        assertTrue("isolated output must not be uniform", isNotUniform(isolated))
    }

    private fun generateWith(
        mode: DiffusionWorkerMode,
        request: ImageGenerationRequest,
    ): Bitmap {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val parentScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val edgeScope = LLMEdgeScope(parentScope, 4)
        val config =
            LLMEdgeConfig(
                image =
                    ImageRuntimeConfig(
                        useVulkan = false,
                        workerMode = mode,
                    ),
            )
        val engine: DiffusionEngine =
            when (mode) {
                DiffusionWorkerMode.IN_PROCESS ->
                    InProcessDiffusionEngine(context, edgeScope, config, DefaultModelRepository())
                DiffusionWorkerMode.ISOLATED_PROCESS ->
                    IsolatedDiffusionEngine(context, edgeScope, config)
            }
        return try {
            runBlocking {
                withTimeout(15 * 60_000) { engine.generate(request) }
            }
        } finally {
            engine.close()
            edgeScope.close()
            parentScope.cancel()
        }
    }

    private fun isNotUniform(bitmap: Bitmap): Boolean {
        val first = bitmap.getPixel(0, 0)
        for (y in 0 until bitmap.height step 8) {
            for (x in 0 until bitmap.width step 8) {
                if (bitmap.getPixel(x, y) != first) return true
            }
        }
        return false
    }
}
