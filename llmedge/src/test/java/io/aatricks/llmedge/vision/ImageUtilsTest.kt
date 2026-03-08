package io.aatricks.llmedge.vision

import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ImageUtilsTest {
    @Test
    fun `rgbBytesToBitmap maps RGB bytes to pixels`() {
        val bitmap =
            ImageUtils.rgbBytesToBitmap(
                byteArrayOf(
                    0xFF.toByte(), 0x00, 0x00,
                    0x00, 0xFF.toByte(), 0x00,
                ),
                width = 2,
                height = 1,
            )

        assertEquals(0xFFFF0000.toInt(), bitmap.getPixel(0, 0))
        assertEquals(0xFF00FF00.toInt(), bitmap.getPixel(1, 0))
    }

    @Test
    fun `preprocessImage scales down oversized images`() {
        val source = Bitmap.createBitmap(2000, 1000, Bitmap.Config.ARGB_8888)

        val scaled = ImageUtils.preprocessBitmap(source, maxDimension = 1000)

        assertEquals(1000, scaled.width)
        assertEquals(500, scaled.height)
        assertTrue(scaled.width <= 1000)
        assertTrue(scaled.height <= 1000)
    }

    @Test
    fun `source-aware preprocessImage delegates through image source helpers`() = kotlinx.coroutines.runBlocking {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val source = ImageSource.BitmapSource(Bitmap.createBitmap(2000, 1000, Bitmap.Config.ARGB_8888))

        val scaled = ImageUtils.preprocessImage(context, source, maxDimension = 1000)

        assertEquals(1000, scaled.width)
        assertEquals(500, scaled.height)
    }
}