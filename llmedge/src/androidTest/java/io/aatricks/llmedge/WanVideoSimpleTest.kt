package io.aatricks.llmedge

import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams

@LargeTest
@RunWith(AndroidJUnit4::class)
class WanVideoSimpleTest {

    @Test
    fun testWanVideoGenerationWithExistingFiles() {
        runBlocking {
            assumeTrue("Requires arm64 device", Build.SUPPORTED_ABIS.any { it.contains("arm64") })

            assumeTrue("Native library not loaded", StableDiffusion.isNativeLibraryLoaded())

            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val filesDir = context.filesDir.absolutePath

            // Use files already on device
            val modelPath =
                    "$filesDir/hf-models/Comfy-Org_Wan_2.1_ComfyUI_repackaged/main/wan2.1_t2v_1.3B_bf16.safetensors"
            val vaePath =
                    "$filesDir/hf-models/Comfy-Org_Wan_2.1_ComfyUI_repackaged/main/wan_2.1_vae.safetensors"
            val t5xxlPath =
                    "$filesDir/hf-models/city96_umt5-xxl-encoder-gguf/main/umt5-xxl-encoder-Q3_K_S.gguf"

            android.util.Log.i(TAG, "====================================")
            android.util.Log.i(TAG, "Wan 2.1 T2V Full Stack Test")
            android.util.Log.i(TAG, "====================================")

            // Verify files exist; otherwise skip the test on devices without preloaded models
            assumeTrue("Model file not found", File(modelPath).exists())
            assumeTrue("VAE file not found", File(vaePath).exists())
            assumeTrue("T5XXL file not found", File(t5xxlPath).exists())

            android.util.Log.i(TAG, "Model: ${File(modelPath).length() / 1024 / 1024}MB")
            android.util.Log.i(TAG, "VAE: ${File(vaePath).length() / 1024 / 1024}MB")
            android.util.Log.i(TAG, "T5XXL: ${File(t5xxlPath).length() / 1024 / 1024}MB")

            android.util.Log.i(TAG, "Loading with CPU offloading...")
            val start = System.currentTimeMillis()

            val sd =
                    StableDiffusion.load(
                            context = context,
                            modelPath = modelPath,
                            vaePath = vaePath,
                            t5xxlPath = t5xxlPath,
                            nThreads = 4,
                            offloadToCpu = true,
                            keepClipOnCpu = true,
                            keepVaeOnCpu = true,
                    )

            android.util.Log.i(TAG, "Loaded in ${System.currentTimeMillis() - start}ms")

            sd.use {
                assertTrue("Should be video model", it.isVideoModel())
                android.util.Log.i(TAG, "✓ Model is video-capable")

                val params =
                        VideoGenerateParams(
                                prompt = "a cat walking",
                                width = 256,
                                height = 256,
                                videoFrames = 4,
                                steps = 10,
                                cfgScale = 7.0f,
                                seed = 42L,
                                sampleMethod = SampleMethod.EULER_A
                        )

                android.util.Log.i(TAG, "Generating video...")
                val genStart = System.currentTimeMillis()
                val frames = it.txt2vid(params)
                val genTime = System.currentTimeMillis() - genStart

                assertNotNull("Frames null", frames)
                assertEquals("Frame count", 4, frames.size)
                frames.forEachIndexed { i, frame ->
                    assertEquals("Frame $i width", 256, frame.width)
                    assertEquals("Frame $i height", 256, frame.height)
                }

                android.util.Log.i(TAG, "====================================")
                android.util.Log.i(TAG, "✅ SUCCESS!")
                android.util.Log.i(TAG, "Generated ${frames.size} frames in ${genTime / 1000}s")
                android.util.Log.i(TAG, "====================================")

                frames.forEach { it.recycle() }
            }
        }
    }

    companion object {
        private const val TAG = "WanVideoSimple"
    }
}
