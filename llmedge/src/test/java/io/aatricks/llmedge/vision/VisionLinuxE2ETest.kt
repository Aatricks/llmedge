package io.aatricks.llmedge.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdge
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.runtime.SmolLM
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VisionLinuxE2ETest {

    private val visionModelEnv = "LLMEDGE_TEST_VISION_MODEL_PATH"
    private val visionProjectorEnv = "LLMEDGE_TEST_VISION_PROJECTOR_PATH"
    private val libPathEnv = "LLMEDGE_BUILD_NATIVE_LIB_PATH"

    @Before
    fun resetNativeBridges() {
        SmolLM.resetNativeBridgeForTests()
        GGUFReader.resetNativeBridgeForTests()
    }

    @After
    fun tearDown() {
        SmolLM.resetNativeBridgeForTests()
        GGUFReader.resetNativeBridgeForTests()
    }

    private fun resolveFromEnvOrDefault(envName: String, defaultPath: String): String? {
        val explicit = System.getenv(envName) ?: System.getProperty(envName)
        return explicit?.takeIf { it.isNotBlank() } ?: defaultPath.takeIf { File(it).exists() }
    }

    private fun createFixtureBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                val r = (x * 255) / (bitmap.width - 1)
                val g = (y * 255) / (bitmap.height - 1)
                val b = if (x in 64..160 && y in 48..176) 32 else 200
                bitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }
        return bitmap
    }

    @Test
    fun `llmedge vision pipeline runs multimodal inference on linux`() = runBlocking {
        val modelPath =
            resolveFromEnvOrDefault(
                visionModelEnv,
                "models/vision/llava-phi-3-mini-int4.gguf",
            )
        val projectorPath =
            resolveFromEnvOrDefault(
                visionProjectorEnv,
                "models/vision/llava-phi-3-mini-mmproj-f16.gguf",
            )

        println("[VisionLinuxE2ETest] modelPath=$modelPath")
        println("[VisionLinuxE2ETest] projectorPath=$projectorPath")

        Assume.assumeTrue(
            "No vision test model specified in $visionModelEnv and default model is missing",
            !modelPath.isNullOrBlank() && File(modelPath).exists(),
        )
        Assume.assumeTrue(
            "No vision projector specified in $visionProjectorEnv and default projector is missing",
            !projectorPath.isNullOrBlank() && File(projectorPath).exists(),
        )

        val libPath =
            System.getenv(libPathEnv)
                ?: System.getProperty(libPathEnv)
                ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsmollm.so"
        Assume.assumeTrue("Native library not found at $libPath", File(libPath).exists())
        Assume.assumeTrue(
            "Native loading is disabled",
            System.getProperty("llmedge.disableNativeLoad") != "true",
        )

        val context = ApplicationProvider.getApplicationContext<Context>()
        val edge =
            LLMEdge.create(
                context = context,
                scope = CoroutineScope(SupervisorJob()),
                config = LLMEdgeConfig(textUseVulkan = false),
            )

        try {
            val bitmap = createFixtureBitmap()
            val response =
                edge.vision.analyze(
                    image = bitmap,
                    prompt = "Describe the image in one short sentence.",
                    model = ModelSpec.localFile(modelPath!!),
                    projector = ModelSpec.localFile(projectorPath!!),
                    numThreads = 2,
                    generationThreads = 1,
                ) { status ->
                    println("[VisionLinuxE2ETest] status=$status")
                }

            println("[VisionLinuxE2ETest] response=$response")
            assertTrue("Vision response should not be blank", response.isNotBlank())

            val memory = edge.vision.getLastRuntimeMemory()
            assertNotNull("Vision runtime memory should be reported", memory)
            assertTrue("Vision runtime native memory should be positive", (memory?.nativeBytes ?: 0L) > 0L)
        } finally {
            edge.close()
        }
    }
}
