package io.aatricks.llmedge

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VisionSmokeTest {

    @Test
    fun ocr_smoke_test_does_not_crash() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val edge = LLMEdge.create(context, this)

        try {
            val bitmap = Bitmap.createBitmap(320, 120, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            canvas.drawColor(Color.WHITE)
            val paint = Paint().apply {
                color = Color.BLACK
                textSize = 42f
                isAntiAlias = true
            }
            canvas.drawText("Vision Smoke", 16f, 72f, paint)

            val text = edge.vision.extractText(bitmap)
            assertNotNull(text)
        } finally {
            edge.close()
        }
    }
}