package io.aatricks.llmedge

import android.graphics.Bitmap
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import io.aatricks.llmedge.image.diffusion.StableDiffusion

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StableDiffusionUpscaleTest {
    private val mockBridge = MockStableDiffusionBridge()

    @Before
    fun setUp() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        StableDiffusion.overrideNativeBridgeForTests { mockBridge }
    }

    @After
    fun tearDown() {
        mockBridge.reset()
        StableDiffusion.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `upscaleImage calls bridge upscale and returns correct Bitmap`() = runTest {
        val inputWidth = 64
        val inputHeight = 64
        val inputBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)

        val factor = 4
        val modelPath = "/path/to/esrgan.bin"
        val nThreads = 4
        val tileSize = 128
        val backend = "cpu"

        val resultBitmap = StableDiffusion.upscaleImage(
            modelPath = modelPath,
            input = inputBitmap,
            factor = factor,
            nThreads = nThreads,
            tileSize = tileSize,
            backend = backend
        )

        assertNotNull(resultBitmap)
        assertEquals(inputWidth * factor, resultBitmap.width)
        assertEquals(inputHeight * factor, resultBitmap.height)

        assertEquals(1, mockBridge.upscaleCalls.size)
        val call = mockBridge.upscaleCalls[0]
        assertEquals(modelPath, call.esrganPath)
        assertEquals(nThreads, call.nThreads)
        assertEquals(tileSize, call.tileSize)
        assertEquals(backend, call.backend)
        assertEquals(inputWidth, call.width)
        assertEquals(inputHeight, call.height)
        assertEquals(factor, call.factor)
    }
}
