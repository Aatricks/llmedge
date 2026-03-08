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

internal object ImagePixelSupport {
    fun rgbBytesToBitmap(
        rgb: ByteArray,
        width: Int,
        height: Int,
        pixels: IntArray? = null,
    ): Bitmap {
        val total = width * height
        val pixelArray = if (pixels == null || pixels.size < total) IntArray(total) else pixels
        var rgbIndex = 0
        var pixelIndex = 0
        while (rgbIndex + 2 < rgb.size && pixelIndex < total) {
            val r = rgb[rgbIndex].toInt() and 0xFF
            val g = rgb[rgbIndex + 1].toInt() and 0xFF
            val b = rgb[rgbIndex + 2].toInt() and 0xFF
            pixelArray[pixelIndex] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            rgbIndex += 3
            pixelIndex += 1
        }
        return Bitmap.createBitmap(pixelArray, 0, width, width, height, Bitmap.Config.ARGB_8888)
    }
}