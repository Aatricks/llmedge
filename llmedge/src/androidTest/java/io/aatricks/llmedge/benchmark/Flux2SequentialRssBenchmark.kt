package io.aatricks.llmedge.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.image.ImageClient
import io.aatricks.llmedge.image.ImageGenerationRequest
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
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Verifies the FLUX.2 Klein sequential-mode fix: the text encoder must be freed before the
 * DiT loads, so peak RSS ≈ max(encoder, DiT) instead of the sum (~5 GB wall pre-fix).
 *
 * Reads VmHWM (peak resident set) from /proc/self/status around the generation.
 *
 * Run with:
 *   ./gradlew :llmedge:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.benchmark.Flux2SequentialRssBenchmark \
 *     -Pandroid.testInstrumentationRunnerArguments.llmedge.benchmark.flux_dit_path=/data/local/tmp/bonsai-flux2-klein-ternary-q2_k.gguf \
 *     -Pandroid.testInstrumentationRunnerArguments.llmedge.benchmark.flux_encoder_path=/data/local/tmp/Qwen_3_4b-Q3_K_M.gguf \
 *     -Pandroid.testInstrumentationRunnerArguments.llmedge.benchmark.flux_vae_path=/data/local/tmp/flux2-vae.safetensors
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class Flux2SequentialRssBenchmark {

    @Test
    fun sequentialGenerationPeakRss() {
        val args = InstrumentationRegistry.getArguments()
        val ditPathArg = args.getString("llmedge.benchmark.flux_dit_path")
        val encoderPathArg = args.getString("llmedge.benchmark.flux_encoder_path")
        val vaePathArg = args.getString("llmedge.benchmark.flux_vae_path")
        assumeTrue(
            "Provide flux_dit_path/flux_encoder_path/flux_vae_path instrumentation args",
            ditPathArg != null && encoderPathArg != null && vaePathArg != null,
        )
        val ditPath = ditPathArg!!
        val encoderPath = encoderPathArg!!
        val vaePath = vaePathArg!!
        assumeTrue(
            "Model files missing on device",
            File(ditPath).exists() && File(encoderPath).exists() && File(vaePath).exists(),
        )

        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        // Put this process in the TOP oom bucket for the duration of the multi-GB loads;
        // as a cached process it gets LMK-killed mid-encoder-load on Samsung devices.
        val keepAlive =
            instrumentation.startActivitySync(
                android.content.Intent(context, KeepAliveActivity::class.java)
                    .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK),
            )
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        @Suppress("DEPRECATION")
        val client = ImageClient.create(context, scope)

        val baselineMb = readVmHwmMb()
        BenchmarkReporter.record("flux2_sequential", "baseline_vmhwm", baselineMb, "MB")

        try {
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

            val bitmap = runBlocking { client.generate(request) }
            assertNotNull("Bitmap should not be null", bitmap)

            val peakMb = readVmHwmMb()
            BenchmarkReporter.record("flux2_sequential", "peak_vmhwm", peakMb, "MB")
            BenchmarkReporter.record("flux2_sequential", "generation_delta", peakMb - baselineMb, "MB")
            BenchmarkReporter.printSummary()
            println("[Flux2Rss] baseline=${baselineMb}MB peak=${peakMb}MB")

            // Encoder ~2.1 GB + DiT ~1.4 GB + VAE + compute: resident-at-once (pre-fix) blew
            // past 4 GB; sequential with the invalidate must stay clearly below.
            assertTrue(
                "Peak RSS ${peakMb}MB suggests encoder+DiT were resident together",
                peakMb < 4096.0,
            )
        } finally {
            client.close()
            scope.cancel()
            keepAlive.finish()
        }
    }

    private fun readVmHwmMb(): Double {
        val line = File("/proc/self/status").readLines().firstOrNull { it.startsWith("VmHWM") }
            ?: return 0.0
        val kb = line.split(Regex("\\s+")).getOrNull(1)?.toLongOrNull() ?: return 0.0
        return kb / 1024.0
    }
}
