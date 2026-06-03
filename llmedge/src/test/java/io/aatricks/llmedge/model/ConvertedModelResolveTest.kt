package io.aatricks.llmedge.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.core.LLMEdgeException
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
class ConvertedModelResolveTest {
    @Test
    fun `resolve throws actionable error when converted gguf is missing`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val spec =
            ModelSpec.safetensors(
                "deepgrove/Bonsai",
                ConversionPrecision.IQ2_BN,
                ConversionAdapter.BONSAI_QLINEAR,
            )
        try {
            DefaultModelRepository().resolve(context, spec)
            fail("expected LLMEdgeException for un-converted safetensors spec")
        } catch (e: LLMEdgeException) {
            val msg = e.message ?: ""
            assertTrue(msg, msg.contains("safetensors-convert"))
            assertTrue(msg, msg.contains("--adapter bonsai-qlinear"))
            assertTrue(msg, msg.contains("--precision iq2_bn"))
        }
    }

    @Test
    fun `resolve returns the cached converted gguf when present`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val spec = ModelSpec.safetensorsLocal("/models/bonsai", ConversionPrecision.F16)
        val target = convertedModelTarget(context, spec, spec.hints.conversion!!)
        target.parentFile!!.mkdirs()
        target.writeBytes(byteArrayOf(0x01))
        try {
            val resolved = DefaultModelRepository().resolve(context, spec)
            assertEquals(target.absolutePath, resolved.absolutePath)
        } finally {
            target.delete()
        }
    }
}
