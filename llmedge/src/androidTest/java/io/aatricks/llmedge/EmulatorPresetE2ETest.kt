package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelChatTemplates
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.TextModelOptions
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device E2E for the low-end presets (BitNet text + SmolVLM2 vision). Provision the GGUFs into the
 * test app's external files dir (adb push to /sdcard/Android/data/io.aatricks.llmedge.test/files/);
 * tests skip via [Assume] if absent.
 *
 * Intended for a real arm64 device. NOTE: on the Android emulator running on Apple-Silicon hosts, the
 * inference thread hits SIGILL on a pointer-authentication (PAC `autia`) instruction inside the NDK's
 * prebuilt libc++ — a known HVF PAC-virtualization quirk, not a code bug (the same build generates
 * correct output on real arm64; verified on the desktop host JNI build → "Paris"). To run on that
 * emulator, the native libs must be built without PAC (incl. a PAC-free libc++) or PAC disabled in the
 * guest kernel.
 */
@RunWith(AndroidJUnit4::class)
class EmulatorPresetE2ETest {
    private val ctx = ApplicationProvider.getApplicationContext<Context>()
    private fun modelDir() = ctx.getExternalFilesDir(null)!!

    @Test
    fun bitnet_generates_paris_on_device() = runBlocking {
        val model = File(modelDir(), "bitnet1582b4t-iq2_bn.gguf")
        Assume.assumeTrue("BitNet not provisioned at ${model.absolutePath}", model.exists())

        val edge = LLMEdge.create(ctx, CoroutineScope(Dispatchers.Default))
        try {
            // localFile + the preset's chat template hint exercises the Track A template threading.
            val spec = ModelSpec.localFile(
                model.absolutePath,
                ModelHints(
                    artifactKind = ModelArtifactKind.GGUF_MODEL,
                    capabilities = setOf(ModelCapability.TEXT),
                    chatTemplate = ModelChatTemplates.BITNET,
                ),
            )
            val resp = edge.text.generate(
                prompt = "What is the capital of France? Answer in one word.",
                model = spec,
                // useMmap=false: read the GGUF into RAM instead of mmap over the emulator's FUSE
                // /sdcard (large-file mmap there destabilizes the FuseDaemon). 6 GB AVD has room.
                options = TextModelOptions(temperature = 0.0f, contextSize = 1024L, useMmap = false),
                maxTokens = 16,
            )
            android.util.Log.i("EmulatorPresetE2ETest", "BitNet -> '$resp'")
            val cleaned = resp.replace(Regex("<\\|[^|]*\\|>"), "").trim()
            assertTrue("Expected coherent text, got '$resp'", cleaned.length >= 3)
            assertTrue("Expected 'Paris' in '$resp'", cleaned.contains("Paris", ignoreCase = true))
        } finally {
            edge.close()
        }
    }

    @Test
    fun smolvlm2_describes_image_on_device() = runBlocking {
        val base = File(modelDir(), "SmolVLM2-256M-Video-Instruct-Q8_0.gguf")
        val proj = File(modelDir(), "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf")
        Assume.assumeTrue("SmolVLM2 not provisioned", base.exists() && proj.exists())

        val edge = LLMEdge.create(ctx, CoroutineScope(Dispatchers.Default))
        try {
            val bmp = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888).also {
                val c = Canvas(it)
                c.drawColor(Color.WHITE)
                c.drawCircle(64f, 64f, 40f, Paint().apply { color = Color.RED })
            }
            val desc = edge.vision.analyze(
                image = bmp,
                prompt = "Describe this image briefly.",
                model = ModelSpec.localFile(
                    base.absolutePath,
                    ModelHints(
                        artifactKind = ModelArtifactKind.GGUF_MODEL,
                        capabilities = setOf(ModelCapability.TEXT, ModelCapability.VISION),
                    ),
                ),
                projector = ModelSpec.localFile(
                    proj.absolutePath,
                    ModelHints(
                        artifactKind = ModelArtifactKind.PROJECTOR,
                        capabilities = setOf(ModelCapability.PROJECTOR),
                    ),
                ),
            )
            android.util.Log.i("EmulatorPresetE2ETest", "SmolVLM2 -> '$desc'")
            assertTrue("Expected non-empty vision description, got '$desc'", desc.trim().length >= 3)
        } finally {
            edge.close()
        }
    }
}
