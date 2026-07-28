package io.aatricks.llmedge.model

import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.core.InvalidModelFileException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Runs the all-in-one classification on device, through the packaged `libggufreader.so` — the
 * build that actually ships, rather than the desktop JNI library the host-side test uses.
 *
 * Fixtures are header-only range reads of the published SD3 Medium checkpoints; the classifier
 * never touches weights, so 8 MB is enough. Push them before running:
 *
 * ```
 * curl -L -r 0-8388607 -o bundle.gguf \
 *   https://huggingface.co/second-state/stable-diffusion-3-medium-GGUF/resolve/main/sd3-medium-Q4_0.gguf
 * curl -L -r 0-8388607 -o dit.gguf \
 *   https://huggingface.co/city96/stable-diffusion-3-medium-gguf/resolve/main/sd3_medium-Q4_0.gguf
 * adb push bundle.gguf dit.gguf /sdcard/Android/data/io.aatricks.llmedge.test/files/
 * ```
 */
class GgufClassificationDeviceTest {
    private val fixtures: File?
        get() =
            InstrumentationRegistry.getInstrumentation()
                .targetContext
                .getExternalFilesDir(null)

    @Test
    fun publishedAllInOneCheckpointIsRejectedForADiffusionOnlySlot() {
        val file = fixture("bundle.gguf")

        val summary = requireNotNull(GgufFileSummary.read(file)) { "native reader returned no summary" }
        android.util.Log.i(TAG, "bundle prefixes=${summary.tensorPrefixes.sorted()} components=${summary.components}")
        assertTrue(summary.isAllInOne)

        try {
            ModelFileValidator.requireDiffusionOnlyGguf(file, "sd3-medium-Q4_0.gguf")
            fail("Expected the bundle to be rejected")
        } catch (expected: InvalidModelFileException) {
            android.util.Log.i(TAG, "rejection message: ${expected.message}")
            assertTrue(expected.message.orEmpty().contains("all-in-one"))
        }
    }

    @Test
    fun publishedDiffusionOnlyCheckpointIsAccepted() {
        val file = fixture("dit.gguf")

        val summary = requireNotNull(GgufFileSummary.read(file)) { "native reader returned no summary" }
        android.util.Log.i(TAG, "dit prefixes=${summary.tensorPrefixes.sorted()} components=${summary.components}")
        assertFalse(summary.isAllInOne)
        assertTrue(GgufComponent.DIFFUSION in summary.components)
        assertEquals(file, ModelFileValidator.requireDiffusionOnlyGguf(file))
    }

    private fun fixture(name: String): File {
        val file = File(fixtures, name)
        assumeTrue("Missing fixture ${file.absolutePath}; see the KDoc for how to push it", file.isFile)
        return file
    }

    private companion object {
        const val TAG = "GgufClassificationDeviceTest"
    }
}
