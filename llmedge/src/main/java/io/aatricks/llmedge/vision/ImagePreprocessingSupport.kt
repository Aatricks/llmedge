/*
 * Copyright (C) 2024 Aatricks
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.aatricks.llmedge.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint

internal object ImagePreprocessingSupport {
    const val MAX_DIMENSION = 1600

    fun preprocessBitmap(
        bitmap: Bitmap,
        maxDimension: Int,
        enhance: Boolean,
    ): Bitmap {
        var result = bitmap
        if (bitmap.width > maxDimension || bitmap.height > maxDimension) {
            result = scaleBitmap(result, maxDimension)
        }
        if (enhance) {
            result = enhanceForOcr(result)
        }
        return result
    }

    private fun scaleBitmap(bitmap: Bitmap, maxDimension: Int): Bitmap {
        if (bitmap.width <= maxDimension && bitmap.height <= maxDimension) {
            return bitmap
        }

        val scale =
            if (bitmap.width > bitmap.height) {
                maxDimension.toFloat() / bitmap.width
            } else {
                maxDimension.toFloat() / bitmap.height
            }

        return Bitmap.createScaledBitmap(
            bitmap,
            (bitmap.width * scale).toInt(),
            (bitmap.height * scale).toInt(),
            true,
        )
    }

    private fun enhanceForOcr(bitmap: Bitmap): Bitmap {
        val grayscale = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscale)
        val paint = Paint()

        val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
        val contrastMatrix =
            ColorMatrix(
                floatArrayOf(
                    1.5f, 0f, 0f, 0f, -40f,
                    0f, 1.5f, 0f, 0f, -40f,
                    0f, 0f, 1.5f, 0f, -40f,
                    0f, 0f, 0f, 1f, 0f,
                ),
            )
        colorMatrix.postConcat(contrastMatrix)
        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return grayscale
    }
}