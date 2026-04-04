package io.aatricks.llmedge.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.core.ProgressEvent
import java.io.FileNotFoundException
import java.nio.file.Files
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class DefaultModelRepositoryTest {
    @Test
    fun `resolve returns local file unchanged`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val file =
            Files.createTempFile("llmedge", ".gguf").toFile().apply {
                writeBytes(byteArrayOf(0x01))
            }

        try {
            val resolver = DefaultModelRepository()
            val resolved = resolver.resolve(context, ModelSpec.localFile(file))
            assertEquals(file.absolutePath, resolved.absolutePath)
        } finally {
            file.delete()
        }
    }

    @Test
    fun `resolve throws when local file is missing`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val file = Files.createTempDirectory("llmedge").resolve("missing.gguf").toFile()
        val resolver = DefaultModelRepository()

        try {
            resolver.resolve(context, ModelSpec.localFile(file))
            fail("Expected FileNotFoundException")
        } catch (expected: FileNotFoundException) {
            assertTrue(expected.message.orEmpty().contains("Model file not found"))
        }
    }

    @Test
    fun `resolve propagates download progress for Hugging Face models`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val file =
            Files.createTempFile("llmedge", ".gguf").toFile().apply {
                writeBytes(byteArrayOf(0x01))
            }
        val events = mutableListOf<ProgressEvent.Downloading>()
        val resolver =
            DefaultModelRepository(
                huggingFaceResolver = { _, spec, onProgress ->
                    file.also {
                        onProgress?.invoke(
                            ProgressEvent.Downloading(
                                model = spec,
                                downloadedBytes = 5L,
                                totalBytes = 10L,
                            ),
                        )
                    }
                },
            )

        try {
            val spec = ModelSpec.huggingFace(repoId = "repo/model", filename = "model.gguf") as ModelSpec.HuggingFace
            val resolved = resolver.resolve(context, spec) { events += it }

            assertEquals(file.absolutePath, resolved.absolutePath)
            assertEquals(1, events.size)
            assertEquals(spec, events.single().model)
            assertEquals(5L, events.single().downloadedBytes)
            assertEquals(10L, events.single().totalBytes)
        } finally {
            file.delete()
        }
    }
}
