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
import java.io.OutputStream

internal class GifEncoder {
    private var width = 0
    private var height = 0
    private var out: OutputStream? = null
    private var pixels: ByteArray? = null
    private var indexedPixels: ByteArray? = null
    private var colorDepth = 8
    private var colorTab: ByteArray? = null
    private var palSize = 7
    private var dispose = -1
    private var repeatCount = -1
    private var delay = 0
    private var sample = 10
    private var quantizer: GifColorQuantizer? = null

    fun setDelay(ms: Int) {
        delay = ms / 10
    }

    fun setRepeat(iter: Int) {
        repeatCount = iter
    }

    fun setQuality(quality: Int) {
        sample = quality.coerceIn(1, 30)
    }

    fun setDispose(disposalMode: Int) {
        dispose = disposalMode
    }

    fun start(os: OutputStream) {
        out = os
        writeString("GIF89a")
    }

    fun addFrame(image: Bitmap, isFirst: Boolean): Boolean {
        if (out == null) return false

        width = image.width
        height = image.height
        capturePixels(image)

        if (isFirst) {
            analyzePixels()
            writeLogicalScreenDescriptor()
            writePalette()
            if (repeatCount >= 0) writeNetscapeExtension()
        } else {
            indexPixelsWithExistingPalette()
        }

        writeGraphicControlExtension()
        writeImageDescriptor()
        writePixels()
        return true
    }

    fun finish(): Boolean {
        out?.write(0x3b)
        out?.flush()
        return true
    }

    private fun capturePixels(image: Bitmap) {
        pixels = ByteArray(image.width * image.height * 3)
        val data = IntArray(image.width * image.height)
        image.getPixels(data, 0, image.width, 0, 0, image.width, image.height)

        for (index in data.indices) {
            val pixel = data[index]
            val offset = index * 3
            pixels!![offset] = (pixel and 0xff).toByte()
            pixels!![offset + 1] = ((pixel shr 8) and 0xff).toByte()
            pixels!![offset + 2] = ((pixel shr 16) and 0xff).toByte()
        }
    }

    private fun analyzePixels() {
        val pixelBytes = pixels ?: return
        val pixelCount = pixelBytes.size / 3
        indexedPixels = ByteArray(pixelCount)

        quantizer = GifColorQuantizer(pixelBytes, pixelBytes.size, sample)
        colorTab = quantizer!!.process()

        for (index in 0 until pixelCount) {
            val paletteIndex =
                quantizer!!.map(
                    pixelBytes[index * 3].toInt() and 0xff,
                    pixelBytes[index * 3 + 1].toInt() and 0xff,
                    pixelBytes[index * 3 + 2].toInt() and 0xff,
                )
            indexedPixels!![index] = paletteIndex.toByte()
        }
        pixels = null
        colorDepth = 8
        palSize = 7
    }

    private fun indexPixelsWithExistingPalette() {
        val pixelBytes = pixels ?: return
        val pixelCount = pixelBytes.size / 3
        indexedPixels = ByteArray(pixelCount)
        val activeQuantizer = quantizer ?: return
        for (index in 0 until pixelCount) {
            val paletteIndex =
                activeQuantizer.map(
                    pixelBytes[index * 3].toInt() and 0xff,
                    pixelBytes[index * 3 + 1].toInt() and 0xff,
                    pixelBytes[index * 3 + 2].toInt() and 0xff,
                )
            indexedPixels!![index] = paletteIndex.toByte()
        }
        pixels = null
    }

    private fun writeLogicalScreenDescriptor() {
        writeShort(width)
        writeShort(height)
        out?.write(0x80 or 0x70 or palSize)
        out?.write(0)
        out?.write(0)
    }

    private fun writeNetscapeExtension() {
        out?.write(0x21)
        out?.write(0xff)
        out?.write(11)
        writeString("NETSCAPE2.0")
        out?.write(3)
        out?.write(1)
        writeShort(repeatCount)
        out?.write(0)
    }

    private fun writePalette() {
        val palette = colorTab ?: return
        out?.write(palette, 0, palette.size)
        repeat(3 * 256 - palette.size) { out?.write(0) }
    }

    private fun writeGraphicControlExtension() {
        out?.write(0x21)
        out?.write(0xf9)
        out?.write(4)
        val disp = if (dispose >= 0) dispose shl 2 else 0
        out?.write(disp)
        writeShort(delay)
        out?.write(0)
        out?.write(0)
    }

    private fun writeImageDescriptor() {
        out?.write(0x2c)
        writeShort(0)
        writeShort(0)
        writeShort(width)
        writeShort(height)
        out?.write(0)
    }

    private fun writePixels() {
        GifLzwEncoder(indexedPixels!!, colorDepth).encode(out!!)
    }

    private fun writeShort(value: Int) {
        out?.write(value and 0xff)
        out?.write((value shr 8) and 0xff)
    }

    private fun writeString(value: String) {
        for (character in value) out?.write(character.code)
    }
}
