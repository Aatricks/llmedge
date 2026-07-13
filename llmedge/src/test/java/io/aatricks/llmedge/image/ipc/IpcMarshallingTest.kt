package io.aatricks.llmedge.image.ipc

import android.os.Parcel
import android.os.Parcelable
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageRequestMetrics
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelConversion
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class IpcMarshallingTest {
    private inline fun <reified T : Parcelable> parcelRoundTrip(value: T): T {
        val parcel = Parcel.obtain()
        return try {
            parcel.writeParcelable(value, 0)
            parcel.setDataPosition(0)
            @Suppress("DEPRECATION")
            parcel.readParcelable(T::class.java.classLoader)!!
        } finally {
            parcel.recycle()
        }
    }

    @Test
    fun `local file model spec survives codec and parcel round trip`() {
        val spec =
            ModelSpec.LocalFile(
                file = File("/data/local/tmp/model.gguf"),
                hints =
                    ModelHints(
                        capabilities = setOf(ModelCapability.IMAGE),
                        conversion = ModelConversion(tokenizerPre = "smollm"),
                    ),
            )
        val decoded = IpcCodecs.fromIpc(parcelRoundTrip(IpcCodecs.toIpc(spec)))
        assertEquals(spec, decoded)
    }

    @Test
    fun `hugging face model spec survives codec and parcel round trip`() {
        val spec =
            ModelSpec.HuggingFace(
                repoId = "Green-Sky/SD-Turbo-GGUF",
                filename = "sd_turbo-f16-q8_0.gguf",
                revision = "main",
                preferredQuantizations = listOf("q8_0", "f16"),
                token = "hf_secret",
                forceDownload = true,
                preferSystemDownloader = false,
                hints = ModelHints(chatTemplate = "tmpl"),
            )
        val decoded = IpcCodecs.fromIpc(parcelRoundTrip(IpcCodecs.toIpc(spec)))
        assertEquals(spec, decoded)
    }

    @Test
    fun `image request survives codec and parcel round trip`() {
        val request =
            ImageGenerationRequest(
                prompt = "a fox",
                negative = "blurry",
                width = 256,
                height = 384,
                steps = 4,
                cfgScale = 1.5f,
                seed = 42L,
                flashAttention = false,
                forceSequentialLoad = true,
                easyCache = EasyCacheParams(enabled = true, reuseThreshold = 0.3f, startPercent = 0.1f, endPercent = 0.9f),
                loraModelDir = "/lora",
                loraApplyMode = LoraApplyMode.AT_RUNTIME,
                model = ModelSpec.LocalFile(File("/m.gguf")),
                diffusionModelOnly = true,
                splitDiffusionModel = true,
                sequential = true,
            )
        val decoded = IpcCodecs.fromIpc(parcelRoundTrip(IpcCodecs.toIpc(request)))
        assertEquals(request, decoded)
    }

    @Test
    fun `metrics survive codec and parcel round trip including request metrics`() {
        val metrics =
            GenerationMetrics(
                totalTimeSeconds = 12.5f,
                framesPerSecond = 0.08f,
                timePerStep = 3.1f,
                peakMemoryUsageMb = 1234L,
                vulkanEnabled = true,
                frameConversionTimeSeconds = 0.2f,
            ).withImageRequestMetrics(
                ImageRequestMetrics(
                    runtimeAcquireMs = 5000L,
                    modelLoadMs = 4800L,
                    generateMs = 12000L,
                    cacheHit = false,
                    backend = "VULKAN",
                    flashAttentionEnabled = true,
                    easyCacheEnabled = false,
                    width = 512,
                    height = 512,
                    steps = 4,
                ),
            )
        val decoded = IpcCodecs.fromIpc(parcelRoundTrip(IpcCodecs.toIpc(metrics)))
        assertEquals(metrics, decoded)
        assertEquals(metrics.imageRequestMetrics, decoded.imageRequestMetrics)
    }
}
