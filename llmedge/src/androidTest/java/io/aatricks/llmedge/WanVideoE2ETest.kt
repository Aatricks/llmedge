package io.aatricks.llmedge

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoGenerateParams
import io.aatricks.llmedge.runtime.CpuTopology

@LargeTest
@RunWith(AndroidJUnit4::class)
class WanVideoE2ETest {

    @Test
    fun wanModelGeneratesVideoFrames() = runBlocking {
        assumeTrue("Requires arm64 device", Build.SUPPORTED_ABIS.any { it.contains("arm64") })
        assumeTrue("Native library not loaded", StableDiffusion.isNativeLibraryLoaded())

        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val targetContext = instrumentation.targetContext
        // Ensure critical assets exist in the test APK; otherwise skip this heavy test
        try {
            instrumentation.context.assets.open(WAN_MODEL_ASSET_NAME).close()
            instrumentation.context.assets.open(WAN_VAE_ASSET_NAME).close()
        } catch (t: Throwable) {
            org.junit.Assume.assumeTrue("Required WAN assets missing from test APK", false)
        }
        val wanAssets = ensureWanAssetsOnDisk(targetContext)

        val engine = StableDiffusion.load(
            context = targetContext,
            modelPath = wanAssets.model.absolutePath,
            vaePath = wanAssets.vae.absolutePath,
            t5xxlPath = wanAssets.textEncoder.absolutePath,
            nThreads = io.aatricks.llmedge.runtime.CpuTopology.getOptimalThreadCount(io.aatricks.llmedge.runtime.CpuTopology.TaskType.DIFFUSION).coerceAtMost(4),
            offloadToCpu = false,
            keepClipOnCpu = false,
            keepVaeOnCpu = false,
        )

        engine.use { sd ->
            assertTrue("Wan model should be detected as video-capable", sd.isVideoModel())
            val params = VideoGenerateParams(
                prompt = "a cinematic shot of a friendly robot waving",
                width = 256,
                height = 256,
                videoFrames = 4,
                steps = 10,
                seed = 42L,
            )

            val frames = sd.txt2vid(params)
            assertEquals("Unexpected frame count", params.videoFrames, frames.size)
            frames.forEach { bitmap ->
                assertBitmapDimensions(bitmap, params)
            }
        }
    }

    private suspend fun ensureWanAssetsOnDisk(context: Context): WanAssetBundle = withContext(Dispatchers.IO) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val assetManager = instrumentation.context.assets
        val modelsDir = File(context.filesDir, "wan-models").apply { if (!exists()) mkdirs() }

        val model = copyAssetIfNeeded(assetManager, modelsDir, WAN_MODEL_ASSET_NAME)
        val vae = copyAssetIfNeeded(assetManager, modelsDir, WAN_VAE_ASSET_NAME)
        val textEncoder = ensureTextEncoderOnDisk(context)

        WanAssetBundle(model = model, vae = vae, textEncoder = textEncoder)
    }

    private fun copyAssetIfNeeded(assetManager: AssetManager, destDir: File, assetName: String): File {
        val destFile = File(destDir, assetName)
        if (!destFile.exists() || destFile.length() == 0L) {
            assetManager.open(assetName).use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        check(destFile.exists() && destFile.length() > 0L) { "Failed to stage asset $assetName" }
        destFile.setReadable(true, false)
        return destFile
    }

    private suspend fun ensureTextEncoderOnDisk(context: Context): File = withContext(Dispatchers.IO) {
        // Prefer in-app HTTP downloader in tests (DownloadManager may not be available on AVDs)
        val preferSystem = false
        val result = try {
            HuggingFaceHub.ensureRepoFileOnDisk(
                context = context,
                modelId = WAN_T5_MODEL_ID,
                revision = "main",
                filename = WAN_T5_FILENAME,
                allowedExtensions = listOf(".gguf"),
                preferSystemDownloader = preferSystem,
            )
        } catch (t: Throwable) {
            // Fall back to system downloader if in-app fails (non-emulator environment)
            HuggingFaceHub.ensureRepoFileOnDisk(
                context = context,
                modelId = WAN_T5_MODEL_ID,
                revision = "main",
                filename = WAN_T5_FILENAME,
                allowedExtensions = listOf(".gguf"),
                preferSystemDownloader = true,
            )
        }
        result.file.setReadable(true, false)
        result.file
    }

    private fun assertBitmapDimensions(bitmap: Bitmap, params: VideoGenerateParams) {
        assertEquals(params.width, bitmap.width)
        assertEquals(params.height, bitmap.height)
    }

    private companion object {
        const val WAN_MODEL_ASSET_NAME = "Wan2.1-T2V-1.3B-Q3_K_S.gguf"
        const val WAN_VAE_ASSET_NAME = "wan_2.1_vae.safetensors"
        const val WAN_T5_MODEL_ID = "city96/umt5-xxl-encoder-gguf"
        const val WAN_T5_FILENAME = "umt5-xxl-encoder-Q3_K_S.gguf"
    }

    private data class WanAssetBundle(
        val model: File,
        val vae: File,
        val textEncoder: File,
    )
}
