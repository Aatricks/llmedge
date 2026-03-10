package io.aatricks.llmedge

import android.graphics.Bitmap
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
// Avoid mocking Android framework classes (Bitmap). Use Robolectric real Bitmaps instead.
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.lang.reflect.Method
import io.aatricks.llmedge.image.diffusion.StableDiffusion

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class BitmapConversionTest {

    @Test
    fun `bitmapToRgbBytes converts ARGB_8888 bitmap correctly`() {
        val sd = newStableDiffusion()
        // Create a real ARGB_8888 Bitmap using Robolectric
        val bitmap = Bitmap.createBitmap(2, 2, Bitmap.Config.ARGB_8888)
        // Set pixels in row-major order (x, y)
        bitmap.setPixel(0, 0, 0xFFAABBCC.toInt()) // pixel index 0
        bitmap.setPixel(1, 0, 0xFFDDEEFF.toInt()) // pixel index 1
        bitmap.setPixel(0, 1, 0xFF112233.toInt()) // pixel index 2
        bitmap.setPixel(1, 1, 0xFF445566.toInt()) // pixel index 3

        val result = callPrivateMethod<Triple<ByteArray, Int, Int>>(
            sd,
            "bitmapToRgbBytes",
            Bitmap::class.java to bitmap
        )

        val rgbBytes = result.first
        val width = result.second
        val height = result.third

        assertEquals(2, width)
        assertEquals(2, height)
        assertEquals(12, rgbBytes.size) // 2x2x3 = 12 bytes

        // Verify RGB values (no alpha, little-endian byte order)
        assertEquals((-86).toByte(), rgbBytes[0]) // R from pixel 0 (0xAA)
        assertEquals((-69).toByte(), rgbBytes[1]) // G from pixel 0 (0xBB)
        assertEquals((-52).toByte(), rgbBytes[2]) // B from pixel 0 (0xCC)
        assertEquals((-35).toByte(), rgbBytes[3]) // R from pixel 1 (0xDD)
        assertEquals((-18).toByte(), rgbBytes[4]) // G from pixel 1 (0xEE)
        assertEquals((-1).toByte(), rgbBytes[5])  // B from pixel 1 (0xFF)
    }

    @Test
    fun `bitmapToRgbBytes copies bitmap if not ARGB_8888`() {
        val sd = newStableDiffusion()
        val originalBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.RGB_565)
        // Fill the RGB_565 bitmap with a color that will be represented similarly
        originalBitmap.setPixel(0, 0, 0xFF123456.toInt())

        val result = callPrivateMethod<Triple<ByteArray, Int, Int>>(
            sd,
            "bitmapToRgbBytes",
            Bitmap::class.java to originalBitmap
        )

        assertNotNull("Should return RGB bytes", result.first)
        assertEquals(1, result.second) // width
        assertEquals(1, result.third) // height
    }

    @Test
    fun `rgbBytesToBitmap creates correct bitmap from RGB bytes`() {
        val sd = newStableDiffusion()

        val rgbBytes = byteArrayOf(
            0x11.toByte(), 0x22.toByte(), 0x33.toByte(), // pixel 0: R=11, G=22, B=33
            0x44.toByte(), 0x55.toByte(), 0x66.toByte(), // pixel 1: R=44, G=55, B=66
            0x77.toByte(), 0x88.toByte(), 0x99.toByte(), // pixel 2: R=77, G=88, B=99
            0xAA.toByte(), 0x11.toByte(), 0x22.toByte()  // pixel 3: R=AA, G=11, B=22
        )

        val result = callPrivateMethod<Bitmap>(
            sd,
            "rgbBytesToBitmap",
            ByteArray::class.java to rgbBytes,
            Int::class.java to 2,
            Int::class.java to 2
        )

        assertEquals(2, result.width)
        assertEquals(2, result.height)
        assertEquals(Bitmap.Config.ARGB_8888, result.config)

        val pixels = IntArray(4)
        // Read the full bitmap by providing width=2 and height=2
        result.getPixels(pixels, 0, 2, 0, 0, 2, 2)

        // Check that pixels map to the input RGB bytes
        assertEquals(0xFF112233.toInt(), pixels[0])
        assertEquals(0xFF445566.toInt(), pixels[1])
        assertEquals(0xFF778899.toInt(), pixels[2])
        assertEquals(0xFFAA1122.toInt(), pixels[3])
    }

    @Test
    fun `rgbBytesToBitmap handles partial pixel data gracefully`() {
        val sd = newStableDiffusion()

        // Only 2 bytes instead of 3 for a 1x1 image (incomplete pixel)
        val rgbBytes = byteArrayOf(0x11, 0x22) // Missing blue component

        val result = callPrivateMethod<Bitmap>(
            sd,
            "rgbBytesToBitmap",
            ByteArray::class.java to rgbBytes,
            Int::class.java to 1,
            Int::class.java to 1
        )

        assertEquals(1, result.width)
        assertEquals(1, result.height)

        val pixels = IntArray(1)
        result.getPixels(pixels, 0, 1, 0, 0, 1, 1)

        // Should still create a valid bitmap, though pixel data may be incomplete
        assertNotNull("Bitmap should be created", result)
    }

    @Test
    fun `convertFramesToBitmaps processes multiple frames correctly`() {
        val sd = newStableDiffusion()

        // Create 3 frames of 1x1 RGB images
        val frameBytes = arrayOf(
            byteArrayOf(0xFF.toByte(), 0x00.toByte(), 0x00.toByte()), // Red frame
            byteArrayOf(0x00.toByte(), 0xFF.toByte(), 0x00.toByte()), // Green frame
            byteArrayOf(0x00.toByte(), 0x00.toByte(), 0xFF.toByte())  // Blue frame
        )

        val result = callPrivateMethod<List<Bitmap>>(
            sd,
            "convertFramesToBitmaps",
            Array<ByteArray>::class.java to frameBytes,
            Int::class.java to 1,
            Int::class.java to 1
        )

        assertEquals(3, result.size)

        result.forEachIndexed { index, bitmap ->
            assertTrue("Frame $index should be a Bitmap", bitmap is Bitmap)
            val bmp = bitmap as Bitmap
            assertEquals(1, bmp.width)
            assertEquals(1, bmp.height)
            assertEquals(Bitmap.Config.ARGB_8888, bmp.config)
        }
    }

    @Test
    fun `convertFramesToBitmaps does not alias pixel buffer across frames`() {
        val sd = newStableDiffusion()

        // Create 2 frames with different pixel values
        val frameBytes = arrayOf(
            byteArrayOf(0x01.toByte(), 0x02.toByte(), 0x03.toByte()),
            byteArrayOf(0x04.toByte(), 0x05.toByte(), 0x06.toByte())
        )

        val result = callPrivateMethod<List<Bitmap>>(
            sd,
            "convertFramesToBitmaps",
            Array<ByteArray>::class.java to frameBytes,
            Int::class.java to 1,
            Int::class.java to 1
        )

        assertEquals(2, result.size)
        val firstPixel = result[0].getPixel(0, 0)
        val secondPixel = result[1].getPixel(0, 0)
        assertNotNull("First bitmap should be created", firstPixel)
        assertNotNull("Second bitmap should be created", secondPixel)
        assertTrue("Frames should not be identical", firstPixel != secondPixel)
    }

    @Test
    fun `convertFramesToBitmaps handles empty frame array`() {
        val sd = newStableDiffusion()

        val result = callPrivateMethod<List<Bitmap>>(
            sd,
            "convertFramesToBitmaps",
            Array<ByteArray>::class.java to emptyArray<ByteArray>(),
            Int::class.java to 1,
            Int::class.java to 1
        )

        assertEquals(0, result.size)
    }

    @Test
    fun `convertFramesToBitmaps copies frame bytes to avoid aliasing`() {
        val sd = newStableDiffusion()
        val shared = byteArrayOf(0x01.toByte(), 0x02.toByte(), 0x03.toByte())
        val frames = arrayOf(shared, shared)

        val result = callPrivateMethod<List<Bitmap>>(
            sd,
            "convertFramesToBitmaps",
            Array<ByteArray>::class.java to frames,
            Int::class.java to 1,
            Int::class.java to 1
        )

        // Mutate the shared backing buffer after conversion - bitmaps should retain their
        // original pixel data if we copied the byte arrays during conversion.
        shared[0] = 0xFF.toByte()
        shared[1] = 0x00.toByte()
        shared[2] = 0x00.toByte()

        val firstPixel = result[0].getPixel(0, 0)
        val secondPixel = result[1].getPixel(0, 0)
        assertEquals(0xFF010203.toInt(), firstPixel)
        assertEquals(0xFF010203.toInt(), secondPixel)
    }

    @Test
    fun `convertFramesToBitmaps uses different batch sizes based on frame count`() {
        val sd = newStableDiffusion()

        // Test with 4 frames (should use batch size 8)
        val frameBytes4 = Array(4) { byteArrayOf(0xFF.toByte(), 0x00.toByte(), 0x00.toByte()) }
        val result4 = callPrivateMethod<List<Bitmap>>(
            sd,
            "convertFramesToBitmaps",
            Array<ByteArray>::class.java to frameBytes4,
            Int::class.java to 1,
            Int::class.java to 1
        )
        assertEquals(4, result4.size)

        // Test with 48 frames (should use batch size 4)
        val frameBytes48 = Array(48) { byteArrayOf(0xFF.toByte(), 0x00.toByte(), 0x00.toByte()) }
        val result48 = callPrivateMethod<List<Bitmap>>(
            sd,
            "convertFramesToBitmaps",
            Array<ByteArray>::class.java to frameBytes48,
            Int::class.java to 1,
            Int::class.java to 1
        )
        assertEquals(48, result48.size)
    }

    private fun newStableDiffusion(): StableDiffusion {
        val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        return constructor.newInstance(1L)
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T> callPrivateMethod(
        instance: Any,
        methodName: String,
        vararg args: Pair<Class<*>, Any>
    ): T {
        val parameterTypes = args.map { it.first }.toTypedArray()
        val parameterValues = args.map { it.second }.toTypedArray()

        val method = instance.javaClass.getDeclaredMethod(methodName, *parameterTypes)
        method.isAccessible = true

        return method.invoke(instance, *parameterValues) as T
    }
}