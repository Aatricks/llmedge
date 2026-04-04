package io.aatricks.llmedge.speech

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdge
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.TextRuntimeConfig
import io.aatricks.llmedge.speech.stt.Whisper
import io.aatricks.llmedge.model.ModelSpec
import java.io.File
import kotlin.math.sin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SpeechClientLinuxE2ETest {

    private val LIB_PATH_ENV = "LLMEDGE_BUILD_WHISPER_LIB_PATH"
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_WHISPER_MODEL_PATH"

    private fun generateTestAudio(durationSeconds: Float = 2.0f): FloatArray {
        val sampleRate = Whisper.SAMPLE_RATE
        val numSamples = (sampleRate * durationSeconds).toInt()
        return FloatArray(numSamples) { index ->
            val t = index.toFloat() / sampleRate
            (sin(2.0 * Math.PI * 440.0 * t) * 0.5f).toFloat()
        }
    }

    @Test
    fun `llmedge speech client transcribes on linux`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        println("[SpeechClientLinuxE2ETest] modelPath=$modelPath")
        Assume.assumeTrue("No whisper test model specified in $MODEL_PATH_ENV", !modelPath.isNullOrBlank())

        val libPath =
            System.getenv(LIB_PATH_ENV)
                ?: System.getProperty(LIB_PATH_ENV)
                ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libwhisper_jni.so"
        Assume.assumeTrue("Native library not found at $libPath", File(libPath).exists())
        Assume.assumeTrue(
            "Native loading is disabled",
            System.getProperty("llmedge.disableNativeLoad") != "true"
        )

        val context = ApplicationProvider.getApplicationContext<Context>()
        val edge =
            LLMEdge.create(
                context = context,
                scope = CoroutineScope(SupervisorJob()),
                config = LLMEdgeConfig(text = TextRuntimeConfig(useVulkan = false)),
            )

        try {
            val samples = generateTestAudio()
            val segments =
                edge.speech.transcribe(
                    audioSamples = samples,
                    model = ModelSpec.localFile(modelPath!!),
                    params = Whisper.TranscribeParams(language = "en", printProgress = true),
                    loadOptions = WhisperLoadOptions(useGpu = false, flashAttention = true),
                )

            println("[SpeechClientLinuxE2ETest] segments=${segments.size}")
            segments.forEachIndexed { index, segment ->
                println(
                    "[SpeechClientLinuxE2ETest] Segment $index: [${segment.startTimeMs}ms - ${segment.endTimeMs}ms] ${segment.text}"
                )
            }

            assertNotNull(segments)
        } finally {
            edge.close()
        }
    }
}
