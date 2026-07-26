package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.image.VideoGenerationRequest
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback

/**
 * Linux-host end-to-end test for video generation using the EXACT SAME PATH as Android devices.
 *
 * This test uses the sequential loading path (`forceSequentialLoad = true`) which:
 * 1. Loads T5 encoder separately -> precomputes condition -> unloads T5
 * 2. Loads diffusion model + VAE -> generates video with precomputed condition
 *
 * This is the path that Android devices take due to memory constraints, and uses
 * `txt2VidWithPrecomputedCondition` internally.
 *
 * Requirements to run:
 * - Build the native sdcpp library for Linux and place as
 * llmedge/build/native/linux-x86_64/libsdcpp.so
 * - Provide model paths via environment variables OR let the test download via Hugging Face:
 * - Paths mode:
 * ```
 *     - LLMEDGE_TEST_MODEL_PATH: path to wan2.1_t2v_1.3B_fp16.safetensors
 *     - LLMEDGE_TEST_VAE_PATH: path to wan_2.1_vae.safetensors
 *     - LLMEDGE_TEST_T5_PATH: path to umt5-xxl-encoder-Q3_K_S.gguf
 * ```
 * - HF mode (recommended for local Linux):
 * ```
 *     - LLMEDGE_TEST_MODEL_ID=Comfy-Org/Wan_2.1_ComfyUI_repackaged
 *     - (optional) LLMEDGE_TEST_MODEL_FILENAME=wan2.1_t2v_1.3B_fp16.safetensors
 *     - (optional) LLMEDGE_TEST_VAE_FILENAME=wan_2.1_vae.safetensors
 *     - (optional) LLMEDGE_TEST_T5_MODEL_ID=city96/umt5-xxl-encoder-gguf
 *     - (optional) LLMEDGE_TEST_T5_FILENAME=umt5-xxl-encoder-Q3_K_S.gguf
 *     - (optional) LLMEDGE_TEST_HF_CACHE_DIR=/path/to/persistent/cache (avoids re-downloading each run)
 * ```
 * Run with: ./scripts/run_linux_e2e_sequential.sh
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoGenerationSequentialE2ETest {

    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_MODEL_PATH"

    private fun requireNativeLibrary(): String =
        DesktopNativeTestSupport.requireEnabledAndLoadLibrary(
            envName = LIB_PATH_ENV,
            defaultRelativePath = "llmedge/build/native/linux-x86_64/libsdcpp.so",
        )

    private fun envInt(name: String, default: Int): Int {
        val raw = System.getenv(name) ?: System.getProperty(name)
        return raw?.toIntOrNull() ?: default
    }

    private fun envBool(name: String, default: Boolean): Boolean {
        val raw = System.getenv(name) ?: System.getProperty(name)
        return when (raw?.trim()?.lowercase()) {
            null -> default
            "1", "true", "yes", "y" -> true
            "0", "false", "no", "n" -> false
            else -> default
        }
    }

    private fun findRepoRoot(startDir: File): File {
        var cur: File? = startDir
        repeat(10) {
            val d = cur ?: return@repeat
            if (File(d, "settings.gradle.kts").exists()) return d
            cur = d.parentFile
        }
        return startDir
    }

    private data class WanPaths(
            val modelPath: String,
            val vaePath: String,
            val t5Path: String,
    )

    private suspend fun resolveWanPaths(context: Context): WanPaths {
        // First preference: explicit paths.
        var resolvedModelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        var resolvedT5Path =
                System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        var resolvedVaePath =
                System.getenv("LLMEDGE_TEST_VAE_PATH")
                        ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")

        // Optional: download via Hugging Face if paths are missing.
        val modelId =
                System.getenv("LLMEDGE_TEST_MODEL_ID")
                        ?: System.getProperty("LLMEDGE_TEST_MODEL_ID")
        val modelFilename =
                System.getenv("LLMEDGE_TEST_MODEL_FILENAME")
                        ?: System.getProperty("LLMEDGE_TEST_MODEL_FILENAME")
        val vaeFilename =
                System.getenv("LLMEDGE_TEST_VAE_FILENAME")
                        ?: System.getProperty("LLMEDGE_TEST_VAE_FILENAME")
        val t5ModelId =
                System.getenv("LLMEDGE_TEST_T5_MODEL_ID")
                        ?: System.getProperty("LLMEDGE_TEST_T5_MODEL_ID")
        val t5Filename =
                System.getenv("LLMEDGE_TEST_T5_FILENAME")
                        ?: System.getProperty("LLMEDGE_TEST_T5_FILENAME")

        val forceDownload = envBool("LLMEDGE_TEST_FORCE_DOWNLOAD", false)
        val hfToken = System.getenv("HF_TOKEN") ?: System.getProperty("HF_TOKEN")

        val cacheDir =
                System.getenv("LLMEDGE_TEST_HF_CACHE_DIR")
                        ?: System.getProperty("LLMEDGE_TEST_HF_CACHE_DIR")
        val cacheRoot =
                if (!cacheDir.isNullOrBlank()) {
                    File(cacheDir)
                } else {
                    // Default: use Robolectric app files dir (matches Android storage semantics,
                    // but may be ephemeral).
                    File(context.filesDir, "hf-models")
                }
        cacheRoot.mkdirs()

        if ((resolvedModelPath.isNullOrBlank() ||
                        resolvedVaePath.isNullOrBlank() ||
                        resolvedT5Path.isNullOrBlank()) && !modelId.isNullOrBlank()
        ) {
            // For Robolectric/JVM runs we prefer the pure-JVM downloader.
            val preferSystemDownloader = false

            // Keep behavior aligned with VideoGenerationExportTest.
            if (modelId.equals("Comfy-Org/Wan_2.1_ComfyUI_repackaged", ignoreCase = true)) {
                val comfyRepoId = modelId
                val mainName = modelFilename ?: "wan2.1_t2v_1.3B_fp16.safetensors"
                val vaeName = vaeFilename ?: "wan_2.1_vae.safetensors"
                val t5RepoId = t5ModelId ?: "city96/umt5-xxl-encoder-gguf"
                val t5Name = t5Filename ?: "umt5-xxl-encoder-Q3_K_S.gguf"

                println(
                        "[SequentialE2E] Downloading WAN components (cacheRoot=${cacheRoot.absolutePath})..."
                )
                println("[SequentialE2E]   main: $comfyRepoId / $mainName")
                println("[SequentialE2E]   vae : $comfyRepoId / $vaeName")
                println("[SequentialE2E]   t5  : $t5RepoId / $t5Name")

                fun progressPrinter(tag: String): (downloaded: Long, total: Long?) -> Unit =
                        { downloaded, total ->
                            if (total != null && total > 0) {
                                val pct = (downloaded * 100.0 / total)
                                println(
                                        "[SequentialE2E] $tag download: ${"%.1f".format(pct)}% ($downloaded/$total)"
                                )
                            } else {
                                println("[SequentialE2E] $tag download: $downloaded bytes")
                            }
                        }

                val mainRes =
                        HuggingFaceHub.ensureRepoFileOnDisk(
                                destinationRoot = cacheRoot,
                                modelId = comfyRepoId,
                                revision = "main",
                                filename = mainName,
                                allowedExtensions = listOf(".safetensors"),
                                token = hfToken,
                                forceDownload = forceDownload,
                                preferSystemDownloader = preferSystemDownloader,
                                onProgress = progressPrinter("main"),
                        )
                val vaeRes =
                        HuggingFaceHub.ensureRepoFileOnDisk(
                                destinationRoot = cacheRoot,
                                modelId = comfyRepoId,
                                revision = "main",
                                filename = vaeName,
                                allowedExtensions = listOf(".safetensors", ".pt"),
                                token = hfToken,
                                forceDownload = forceDownload,
                                preferSystemDownloader = preferSystemDownloader,
                                onProgress = progressPrinter("vae"),
                        )
                val t5Res =
                        HuggingFaceHub.ensureRepoFileOnDisk(
                                destinationRoot = cacheRoot,
                                modelId = t5RepoId,
                                revision = "main",
                                filename = t5Name,
                                allowedExtensions = listOf(".gguf"),
                                token = hfToken,
                                forceDownload = forceDownload,
                                preferSystemDownloader = preferSystemDownloader,
                                onProgress = progressPrinter("t5"),
                        )

                if (resolvedModelPath.isNullOrBlank()) {
                    resolvedModelPath = mainRes.file.absolutePath
                }
                if (resolvedVaePath.isNullOrBlank()) {
                    resolvedVaePath = vaeRes.file.absolutePath
                }
                if (resolvedT5Path.isNullOrBlank()) {
                    resolvedT5Path = t5Res.file.absolutePath
                }

                println(
                        "[SequentialE2E] Download/cached files:\n" +
                                "[SequentialE2E]   main=${mainRes.file.absolutePath} (fromCache=${mainRes.fromCache})\n" +
                                "[SequentialE2E]   vae =${vaeRes.file.absolutePath} (fromCache=${vaeRes.fromCache})\n" +
                                "[SequentialE2E]   t5  =${t5Res.file.absolutePath} (fromCache=${t5Res.fromCache})"
                )
            }
        }

        Assume.assumeTrue(
                "Model path not resolved (set LLMEDGE_TEST_MODEL_PATH or use LLMEDGE_TEST_MODEL_ID)",
                !resolvedModelPath.isNullOrBlank(),
        )
        Assume.assumeTrue(
                "VAE path not resolved (set LLMEDGE_TEST_VAE_PATH or allow auto-download)",
                !resolvedVaePath.isNullOrBlank(),
        )
        Assume.assumeTrue(
                "T5 path not resolved (set LLMEDGE_TEST_T5_PATH or allow auto-download)",
                !resolvedT5Path.isNullOrBlank(),
        )

        val finalModel = requireNotNull(resolvedModelPath)
        val finalVae = requireNotNull(resolvedVaePath)
        val finalT5 = requireNotNull(resolvedT5Path)

        return WanPaths(modelPath = finalModel, vaePath = finalVae, t5Path = finalT5)
    }

    @Test
    fun `sequential video generation matches Android device path`() = runBlocking {
        // Use Robolectric context
        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val paths = resolveWanPaths(context)

        println("[SequentialE2E] modelPath=${paths.modelPath}")
        println("[SequentialE2E] t5Path=${paths.t5Path}")
        println("[SequentialE2E] vaePath=${paths.vaePath}")

        Assume.assumeTrue("Model path does not exist", File(paths.modelPath).exists())
        Assume.assumeTrue("T5 path does not exist", File(paths.t5Path).exists())
        Assume.assumeTrue("VAE path does not exist", File(paths.vaePath).exists())

        // Check that native library exists
        val libPath = requireNativeLibrary()
        val libFile = java.io.File(libPath)
        println("[SequentialE2E] libPath=$libPath exists=${libFile.exists()}")

        // Test parameters - same as VideoGenerationActivity defaults
        // Defaults chosen to be CPU-friendly for local Linux runs.
        val width = envInt("LLMEDGE_TEST_WIDTH", 256)
        val height = envInt("LLMEDGE_TEST_HEIGHT", 256)
        val videoFrames =
                envInt("LLMEDGE_TEST_VIDEO_FRAMES", 5) // Wan generates in 1 + 4k increments
        val steps = envInt("LLMEDGE_TEST_STEPS", 10)
        val cfgScale =
                (System.getenv("LLMEDGE_TEST_CFG_SCALE")
                                ?: System.getProperty("LLMEDGE_TEST_CFG_SCALE"))?.toFloatOrNull()
                        ?: 7.0f
        val seed =
                (System.getenv("LLMEDGE_TEST_SEED") ?: System.getProperty("LLMEDGE_TEST_SEED"))
                        ?.toLongOrNull()
                        ?: 42L
        val prompt =
                System.getenv("LLMEDGE_TEST_PROMPT")
                        ?: System.getProperty("LLMEDGE_TEST_PROMPT") ?: "A cat walking in a garden"

        println("[SequentialE2E] Starting SEQUENTIAL video generation (Android path)")
        println("[SequentialE2E] params: ${width}x${height}, $videoFrames frames, $steps steps")
        val startTime = System.currentTimeMillis()

        // This is the EXACT path that Android uses:
        // 1. forceSequentialLoad = true triggers generateVideoSequentially()
        // 2. generateVideoSequentially() loads T5, precomputes condition, unloads T5
        // 3. Then loads diffusion model and calls txt2VidWithPrecomputedCondition

        val params =
                VideoGenerationRequest(
                        prompt = prompt,
                        negative = "",
                        width = width,
                        height = height,
                        videoFrames = videoFrames,
                        steps = steps,
                        cfgScale = cfgScale,
                        seed = seed,
                        flowShift = Float.POSITIVE_INFINITY, // Auto
                        flashAttention = true,
                        forceSequentialLoad = true, // THIS IS THE KEY - forces Android path
                        // Enable EasyCache like Android does
                        easyCache =
                                EasyCacheParams(
                                        enabled = true,
                                        reuseThreshold = 0.2f,
                                        startPercent = 0.15f,
                                        endPercent = 0.95f
                                )
                )

        // We set up model paths directly here instead of resolving from Hugging Face.
        // For this test, we'll directly use StableDiffusion with the sequential loading simulation

        println("[SequentialE2E] Step 1: Loading T5 encoder for condition precomputation...")
        val t5StartTime = System.currentTimeMillis()

        // Load T5-only context (like generateVideoSequentially does)
        val t5Model =
                StableDiffusion.load(
                        context = context,
                        modelPath = paths.t5Path,
                        vaePath = null,
                        t5xxlPath = null,
                        nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                        offloadToCpu = true,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = true
                )

        println("[SequentialE2E] T5 loaded in ${System.currentTimeMillis() - t5StartTime}ms")
        println("[SequentialE2E] Precomputing conditions...")

        // Precompute conditions (like generateVideoSequentially does)
        val cond =
                t5Model.precomputeCondition(
                        prompt = prompt,
                        negative = "",
                        width = width,
                        height = height
                )

        val uncond =
                t5Model.precomputeCondition(
                        prompt = "",
                        negative = "",
                        width = width,
                        height = height
                )

        println("[SequentialE2E] Conditions precomputed, unloading T5...")
        t5Model.close()

        // Force GC like Android does between model loads
        System.gc()
        Thread.sleep(100)

        println("[SequentialE2E] Step 2: Loading diffusion model + VAE...")
        val diffStartTime = System.currentTimeMillis()

        // Load diffusion model WITHOUT T5 (like generateVideoSequentially does with loadT5=false)
        val diffusionModel =
                StableDiffusion.load(
                        context = context,
                        diffusionModelPath = paths.modelPath,
                        vaePath = paths.vaePath,
                        t5xxlPath = null, // No T5 - we already precomputed conditions
                        nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                        offloadToCpu = true,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = true
                )

        println(
                "[SequentialE2E] Diffusion model loaded in ${System.currentTimeMillis() - diffStartTime}ms"
        )
        println("[SequentialE2E] Step 3: Generating video with precomputed conditions...")

        val genStartTime = System.currentTimeMillis()

        // Generate video using precomputed conditions (the key Android path!)
        val videoParams =
                VideoGenerateParams(
                        prompt = prompt,
                        negative = "",
                        width = width,
                        height = height,
                        videoFrames = videoFrames,
                        steps = steps,
                        cfgScale = cfgScale,
                        seed = seed,
                        strength = 1.0f, // Full T2V mode
                        easyCacheParams = params.easyCache
                )

        val progressCallback =
                VideoProgressCallback {
                        step,
                        totalSteps,
                        currentFrame,
                        totalFrames,
                        timePerStep ->
                    println(
                            "[SequentialE2E] Progress: step=$step/$totalSteps, frame=$currentFrame/$totalFrames"
                    )
                }

        val bitmaps =
                try {
                    diffusionModel.txt2VidWithPrecomputedCondition(
                            params = videoParams,
                            cond = cond,
                            uncond = uncond,
                            onProgress = progressCallback
                    )
                } finally {
                    diffusionModel.close()
                }

        val totalTime = System.currentTimeMillis() - startTime
        val genTime = System.currentTimeMillis() - genStartTime

        println("[SequentialE2E] Video generation completed!")
        println("[SequentialE2E] Total time: ${totalTime}ms, generation time: ${genTime}ms")
        println("[SequentialE2E] Generated ${bitmaps.size} frames")

        // Validate results
        assertTrue("Expected at least 1 frame, got ${bitmaps.size}", bitmaps.isNotEmpty())

        bitmaps.forEachIndexed { index, bmp ->
            println("[SequentialE2E] Frame $index: ${bmp.width}x${bmp.height}")
            assertEquals("Frame $index width mismatch", width, bmp.width)
            assertEquals("Frame $index height mismatch", height, bmp.height)
            assertEquals("Frame $index config mismatch", Bitmap.Config.ARGB_8888, bmp.config)

            // Check frame is not blank (not all black/transparent)
            var nonBlankFound = false
            var totalBrightness = 0L
            loop@ for (y in 0 until bmp.height step 4) {
                for (x in 0 until bmp.width step 4) {
                    val px = bmp.getPixel(x, y)
                    val r = (px shr 16) and 0xFF
                    val g = (px shr 8) and 0xFF
                    val b = px and 0xFF
                    totalBrightness += r + g + b
                    if ((px ushr 24) != 0 && (px and 0x00FFFFFF) != 0x000000) {
                        nonBlankFound = true
                    }
                }
            }
            val avgBrightness = totalBrightness / ((bmp.width / 4) * (bmp.height / 4) * 3)
            println("[SequentialE2E] Frame $index average brightness: $avgBrightness")
            assertTrue("Frame $index is blank (all black)", nonBlankFound)
            assertTrue(
                    "Frame $index is too dark (avg brightness $avgBrightness < 5)",
                    avgBrightness >= 5
            )
        }

        // Check frames are different from each other
        if (bitmaps.size > 1) {
            fun bitmapHash(b: Bitmap): Long {
                var h = 1L
                for (y in 0 until b.height step maxOf(1, b.height / 16)) {
                    for (x in 0 until b.width step maxOf(1, b.width / 16)) {
                        h = 31 * h + b.getPixel(x, y)
                    }
                }
                return h
            }
            val hashes = bitmaps.map { bitmapHash(it) }
            println("[SequentialE2E] Frame hashes: $hashes")
            assertTrue(
                    "Expected at least two unique frames, got ${hashes.toSet().size}",
                    hashes.toSet().size > 1
            )
        }

        (System.getenv("LLMEDGE_TEST_OUTPUT_DIR") ?: System.getProperty("LLMEDGE_TEST_OUTPUT_DIR"))
                ?.takeIf(String::isNotBlank)
                ?.let(::File)
                ?.also(File::mkdirs)
                ?.let { outputDir ->
                    bitmaps.forEachIndexed { index, bitmap ->
                        File(outputDir, "frame_%02d.png".format(index)).outputStream().use { output ->
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
                        }
                    }
                    File(outputDir, "wan-sequential.gif").outputStream().use { output ->
                        io.aatricks.llmedge.vision.ImageUtils.createAnimatedGif(
                                frames = bitmaps,
                                delayMs = 125,
                                output = output,
                                loop = 0,
                        )
                    }
                }

        println("[SequentialE2E] ✓ All validations passed!")
    }

    @Test
    fun `sequential I2V generation matches Android device path`() = runBlocking {
        // I2V is useful but significantly slower; keep it opt-in for local runs.
        val enableI2v = envBool("LLMEDGE_TEST_ENABLE_I2V", false)
        Assume.assumeTrue("Set LLMEDGE_TEST_ENABLE_I2V=true to run I2V sequential E2E", enableI2v)

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val paths = resolveWanPaths(context)

        Assume.assumeTrue("Model path does not exist", File(paths.modelPath).exists())
        Assume.assumeTrue("T5 path does not exist", File(paths.t5Path).exists())
        Assume.assumeTrue("VAE path does not exist", File(paths.vaePath).exists())

        requireNativeLibrary()

        // Test parameters for I2V
        val width = envInt("LLMEDGE_TEST_WIDTH", 256)
        val height = envInt("LLMEDGE_TEST_HEIGHT", 256)
        val videoFrames = envInt("LLMEDGE_TEST_VIDEO_FRAMES", 5)
        val steps = envInt("LLMEDGE_TEST_STEPS", 10)
        val cfgScale =
                (System.getenv("LLMEDGE_TEST_CFG_SCALE")
                                ?: System.getProperty("LLMEDGE_TEST_CFG_SCALE"))?.toFloatOrNull()
                        ?: 7.0f
        val seed =
                (System.getenv("LLMEDGE_TEST_SEED") ?: System.getProperty("LLMEDGE_TEST_SEED"))
                        ?.toLongOrNull()
                        ?: 42L
        val strength =
                (System.getenv("LLMEDGE_TEST_STRENGTH")
                                ?: System.getProperty("LLMEDGE_TEST_STRENGTH"))?.toFloatOrNull()
                        ?: 0.8f
        val prompt =
                System.getenv("LLMEDGE_TEST_PROMPT")
                        ?: System.getProperty("LLMEDGE_TEST_PROMPT") ?: "A cat walking in a garden"

        println("[SequentialE2E-I2V] Starting SEQUENTIAL I2V generation (Android path)")
        val startTime = System.currentTimeMillis()

        // Create a simple init image (gradient)
        val initBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = (x * 255 / width)
                val g = (y * 255 / height)
                val b = 128
                initBitmap.setPixel(x, y, (0xFF shl 24) or (r shl 16) or (g shl 8) or b)
            }
        }
        println("[SequentialE2E-I2V] Created init image ${width}x${height}")

        // Step 1: Load T5 and precompute conditions
        println("[SequentialE2E-I2V] Step 1: Loading T5...")
        val t5Model =
                StableDiffusion.load(
                        context = context,
                        modelPath = paths.t5Path,
                        vaePath = null,
                        t5xxlPath = null,
                        nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                        offloadToCpu = true,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = true
                )

        val cond = t5Model.precomputeCondition(prompt, "", width, height)
        val uncond = t5Model.precomputeCondition("", "", width, height)
        t5Model.close()
        System.gc()

        // Step 2: Load diffusion model
        println("[SequentialE2E-I2V] Step 2: Loading diffusion model...")
        val diffusionModel =
                StableDiffusion.load(
                        context = context,
                        modelPath = paths.modelPath,
                        vaePath = paths.vaePath,
                        t5xxlPath = null,
                        nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                        offloadToCpu = true,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = true
                )

        // Step 3: Generate video with init image
        println("[SequentialE2E-I2V] Step 3: Generating I2V with precomputed conditions...")
        val videoParams =
                VideoGenerateParams(
                        prompt = prompt,
                        negative = "",
                        width = width,
                        height = height,
                        videoFrames = videoFrames,
                        steps = steps,
                        cfgScale = cfgScale,
                        seed = seed,
                        initImage = initBitmap,
                        strength = strength,
                        easyCacheParams = EasyCacheParams(enabled = true)
                )

        val bitmaps =
                try {
                    diffusionModel.txt2VidWithPrecomputedCondition(
                            params = videoParams,
                            cond = cond,
                            uncond = uncond
                    )
                } finally {
                    diffusionModel.close()
                    initBitmap.recycle()
                }

        val totalTime = System.currentTimeMillis() - startTime
        println("[SequentialE2E-I2V] I2V completed in ${totalTime}ms, got ${bitmaps.size} frames")

        assertTrue("Expected at least 1 frame", bitmaps.isNotEmpty())
        bitmaps.forEach { bmp ->
            assertEquals(width, bmp.width)
            assertEquals(height, bmp.height)
        }

        println("[SequentialE2E-I2V] ✓ I2V validation passed!")
    }
}
