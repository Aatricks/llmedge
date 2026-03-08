package io.aatricks.llmedge.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
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
class DefaultModelResolverTest {
    @Test
    fun `resolve returns local file unchanged`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val file =
            Files.createTempFile("llmedge", ".gguf").toFile().apply {
                writeBytes(byteArrayOf(0x01))
            }

        try {
            val resolver = DefaultModelResolver()
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
        val resolver = DefaultModelResolver()

        try {
            resolver.resolve(context, ModelSpec.localFile(file))
            fail("Expected FileNotFoundException")
        } catch (expected: FileNotFoundException) {
            assertTrue(expected.message.orEmpty().contains("Model file not found"))
        }
    }
}
