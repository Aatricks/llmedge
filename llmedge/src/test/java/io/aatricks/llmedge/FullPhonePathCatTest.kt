package io.aatricks.llmedge

import android.content.Context
import io.aatricks.llmedge.image.GenerationStreamEvent
import io.aatricks.llmedge.image.VideoGenerationRequest
import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.CoroutineScope
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.flow.collect
import org.junit.Assert.*
import java.io.File

/**
 * High-level E2E test that follows the EXACT same path a phone would,
 * using the instance-based image client.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class FullPhonePathCatTest {

    private fun getModelFile(path: String): File {
        val f1 = File(path)
        if (f1.exists()) return f1
        val f2 = File("..", path)
        if (f2.exists()) return f2
        return f1
    }

    private val MODEL_PATH by lazy { getModelFile("models/wan2.1_t2v_1.3B_fp16.safetensors").absolutePath }
    private val VAE_PATH by lazy { getModelFile("models/wan_2.1_vae.safetensors").absolutePath }
    private val T5_PATH by lazy { getModelFile("models/umt5-xxl-encoder-Q3_K_S.gguf").absolutePath }
    private val TAEHV_PATH by lazy { getModelFile("models/taew2_1.safetensors").absolutePath }

    @Test
    fun `generate cat video using full VAE path`() = runBlocking {
        val testFilter = System.getenv("LLMEDGE_E2E_TEST_TYPE") ?: "all"
        Assume.assumeTrue(testFilter == "all" || testFilter == "vae")
        generateVideoTest(useTaehv = false, "cat_vae.gif")
    }

    @Test
    fun `generate cat video using TAEHV path`() = runBlocking {
        val testFilter = System.getenv("LLMEDGE_E2E_TEST_TYPE") ?: "all"
        Assume.assumeTrue(testFilter == "all" || testFilter == "taehv")
        generateVideoTest(useTaehv = true, "cat_taehv.gif")
    }

    private suspend fun generateVideoTest(useTaehv: Boolean, outputFilename: String) {
        val libPath = System.getenv("LLMEDGE_BUILD_NATIVE_LIB_PATH")
            ?: run {
                val f1 = File("llmedge/build/native/linux-x86_64/libsdcpp.so")
                if (f1.exists()) f1.absolutePath
                else {
                    val f2 = File("../llmedge/build/native/linux-x86_64/libsdcpp.so")
                    if (f2.exists()) f2.absolutePath
                    else f1.absolutePath
                }
            }

        val requiredModelFiles =
            buildList {
                add(File(MODEL_PATH))
                add(File(T5_PATH))
                add(if (useTaehv) File(TAEHV_PATH) else File(VAE_PATH))
            }
        val missingFiles = requiredModelFiles.filterNot(File::exists)
        Assume.assumeTrue(
            "Required local test assets are missing: ${missingFiles.joinToString { it.absolutePath }}",
            missingFiles.isEmpty(),
        )
        
        println("[FullPhonePathCatTest] MODEL_PATH: $MODEL_PATH, exists=${File(MODEL_PATH).exists()}")
        println("[FullPhonePathCatTest] VAE_PATH: $VAE_PATH, exists=${File(VAE_PATH).exists()}")
        println("[FullPhonePathCatTest] T5_PATH: $T5_PATH, exists=${File(T5_PATH).exists()}")
        println("[FullPhonePathCatTest] TAEHV_PATH: $TAEHV_PATH, exists=${File(TAEHV_PATH).exists()}")
        println("[FullPhonePathCatTest] libPath: $libPath, exists=${File(libPath).exists()}")

        Assume.assumeTrue("Native library not found at $libPath", File(libPath).exists())
        
        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val edge =
            LLMEdge.create(
                context,
                CoroutineScope(kotlin.coroutines.coroutineContext),
                config = LLMEdgeConfig(image = ImageRuntimeConfig(preferPerformanceMode = false)),
            )
        val params =
            VideoGenerationRequest(
                prompt = "cat",
                width = 128,
                height = 128,
                videoFrames = 5,
                steps = 1,
                cfgScale = 1.0f,
                seed = 42L,
                forceSequentialLoad = true,
                model = ModelSpec.localFile(MODEL_PATH),
                vae = if (useTaehv) null else ModelSpec.localFile(VAE_PATH),
                textEncoder = ModelSpec.localFile(T5_PATH),
                taehv = if (useTaehv) ModelSpec.localFile(TAEHV_PATH) else null,
            )

        println("[FullPhonePathCatTest] Starting generation: useTaehv=$useTaehv, output=$outputFilename")
        val startTime = System.currentTimeMillis()

        val bitmaps =
            try {
                var completedFrames: List<android.graphics.Bitmap> = emptyList()
                edge.image.generateVideo(params).collect { event ->
                    when (event) {
                        is GenerationStreamEvent.Progress ->
                            println(
                                "[FullPhonePathCatTest] Progress: ${event.update.message} (${event.update.current}/${event.update.total})",
                            )

                        is GenerationStreamEvent.Completed -> completedFrames = event.frames
                    }
                }
                completedFrames
            } catch (e: Exception) {
                println("[FullPhonePathCatTest] Generation failed: ${e.message}")
                e.printStackTrace()
                throw e
            } finally {
                edge.close()
            }

        val elapsed = System.currentTimeMillis() - startTime
        println("[FullPhonePathCatTest] Completed in ${elapsed}ms, got ${bitmaps.size} frames")

        assertTrue("Expected 5 frames (Wan formula (5-1)/4*4+1 = 5), got ${bitmaps.size}", bitmaps.size == 5)

        // Save GIF
        val projectRoot = File(System.getProperty("user.dir") ?: ".")
        val outputGif = File(projectRoot, outputFilename)
        java.io.FileOutputStream(outputGif).use { fos ->
            io.aatricks.llmedge.vision.ImageUtils.createAnimatedGif(
                frames = bitmaps,
                delayMs = 125,
                output = fos,
                loop = 0
            )
        }
        println("[FullPhonePathCatTest] Saved GIF to ${outputGif.absolutePath}")
    }
}
