package io.aatricks.llmedge

import android.content.Context
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ReproductionTest {

    init {
        // Load the native library built for desktop
        val libPath = File("../scripts/jni-desktop/build/libsdcpp.so").absolutePath
        println("Loading native library from: $libPath")
        System.load(libPath)
    }

    @Test
    fun testPublicApiLinking() = runBlocking {
        println("--- STARTING PUBLIC API LINKING TEST ---")
        
        // 1. Test Vulkan query (This often crashed on non-Vulkan devices if linking was bad)
        println("Calling StableDiffusion.getVulkanDeviceCount()...")
        try {
            val count = StableDiffusion.getVulkanDeviceCount()
            println("SUCCESS: Vulkan device count = $count")
        } catch (e: Throwable) {
            println("FAILED: Vulkan query failed: ${e.message}")
            e.printStackTrace()
        }

        // 2. Test Model Loading (This calls nativeCreate)
        val t5Path = File("../models/umt5-xxl-encoder-Q3_K_S.gguf").absolutePath
        if (!File(t5Path).exists()) {
            println("Skipping load test: Missing T5 model at $t5Path")
            return@runBlocking
        }

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        println("\nCalling StableDiffusion.load() with T5 model...")
        try {
            // This calls the public API which internally calls the private nativeCreate
            val sd = StableDiffusion.load(
                context = context,
                modelPath = t5Path,
                vaePath = null,
                t5xxlPath = null,
                nThreads = 4,
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                vaeDecodeOnly = true,
                sequentialLoad = false,
                forceVulkan = false
            )
            println("SUCCESS: StableDiffusion.load() completed successfully.")
            sd.close()
        } catch (e: Throwable) {
            println("FAILED: StableDiffusion.load() failed: ${e.message}")
            e.printStackTrace()
        }
    }
}
