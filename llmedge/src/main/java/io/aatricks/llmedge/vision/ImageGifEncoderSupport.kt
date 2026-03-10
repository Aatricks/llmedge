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

internal object ImageGifEncoderSupport {
    fun createAnimatedGif(
        frames: List<Bitmap>,
        delayMs: Int,
        output: OutputStream,
        loop: Int,
        quality: Int,
    ) {
        if (frames.isEmpty()) return

        val width = frames[0].width
        val height = frames[0].height
        val encoder = GifEncoder().apply {
            setDispose(1)
            setQuality(quality.coerceIn(1, 30))
            start(output)
            setRepeat(loop)
            setDelay(delayMs)
        }

        var first = true
        for (frame in frames) {
            val resized =
                if (frame.width != width || frame.height != height) {
                    Bitmap.createScaledBitmap(frame, width, height, true)
                } else {
                    frame
                }

            encoder.addFrame(resized, isFirst = first)
            first = false

            if (resized !== frame) {
                resized.recycle()
            }
        }

        encoder.finish()
    }

    private class GifEncoder {
        private var width = 0
        private var height = 0
        private var out: OutputStream? = null
        private var pixels: ByteArray? = null
        private var indexedPixels: ByteArray? = null
        private var colorDepth = 8
        private var colorTab: ByteArray? = null
        private var usedEntry = BooleanArray(256)
        private var palSize = 7
        private var dispose = -1
        private var repeat = -1
        private var delay = 0
        private var sample = 10
        private var nq: NeuQuant? = null

        fun setDelay(ms: Int) {
            delay = ms / 10
        }

        fun setRepeat(iter: Int) {
            repeat = iter
        }

        fun setQuality(quality: Int) {
            sample = quality.coerceIn(1, 30)
        }

        fun setDispose(d: Int) {
            dispose = d
        }

        fun start(os: OutputStream) {
            out = os
            writeString("GIF89a")
        }

        fun addFrame(im: Bitmap, isFirst: Boolean): Boolean {
            if (out == null) return false

            width = im.width
            height = im.height
            getImagePixels(im)

            if (isFirst) {
                analyzePixels()
                writeLSD()
                writePalette()
                if (repeat >= 0) writeNetscapeExt()
            } else {
                indexPixelsWithExistingPalette()
            }

            writeGraphicCtrlExt()
            writeImageDesc()
            writePixels()
            return true
        }

        fun finish(): Boolean {
            out?.write(0x3b)
            out?.flush()
            return true
        }

        private fun getImagePixels(image: Bitmap) {
            pixels = ByteArray(image.width * image.height * 3)
            val data = IntArray(image.width * image.height)
            image.getPixels(data, 0, image.width, 0, 0, image.width, image.height)

            for (i in data.indices) {
                val pixel = data[i]
                val offset = i * 3
                pixels!![offset] = (pixel and 0xff).toByte()
                pixels!![offset + 1] = ((pixel shr 8) and 0xff).toByte()
                pixels!![offset + 2] = ((pixel shr 16) and 0xff).toByte()
            }
        }

        private fun analyzePixels() {
            val pixelBytes = pixels ?: return
            val nPix = pixelBytes.size / 3
            indexedPixels = ByteArray(nPix)

            nq = NeuQuant(pixelBytes, pixelBytes.size, sample)
            colorTab = nq!!.process()

            for (i in 0 until nPix) {
                val index =
                    nq!!.map(
                        pixelBytes[i * 3].toInt() and 0xff,
                        pixelBytes[i * 3 + 1].toInt() and 0xff,
                        pixelBytes[i * 3 + 2].toInt() and 0xff,
                    )
                usedEntry[index] = true
                indexedPixels!![i] = index.toByte()
            }
            pixels = null
            colorDepth = 8
            palSize = 7
        }

        private fun indexPixelsWithExistingPalette() {
            val pixelBytes = pixels ?: return
            val nPix = pixelBytes.size / 3
            indexedPixels = ByteArray(nPix)
            val quantizer = nq ?: return
            for (i in 0 until nPix) {
                val index =
                    quantizer.map(
                        pixelBytes[i * 3].toInt() and 0xff,
                        pixelBytes[i * 3 + 1].toInt() and 0xff,
                        pixelBytes[i * 3 + 2].toInt() and 0xff,
                    )
                indexedPixels!![i] = index.toByte()
            }
            pixels = null
        }

        private fun writeLSD() {
            writeShort(width)
            writeShort(height)
            out?.write(0x80 or 0x70 or palSize)
            out?.write(0)
            out?.write(0)
        }

        private fun writeNetscapeExt() {
            out?.write(0x21)
            out?.write(0xff)
            out?.write(11)
            writeString("NETSCAPE2.0")
            out?.write(3)
            out?.write(1)
            writeShort(repeat)
            out?.write(0)
        }

        private fun writePalette() {
            val palette = colorTab ?: return
            out?.write(palette, 0, palette.size)
            repeat(3 * 256 - palette.size) { out?.write(0) }
        }

        private fun writeGraphicCtrlExt() {
            out?.write(0x21)
            out?.write(0xf9)
            out?.write(4)
            val disp = if (dispose >= 0) dispose shl 2 else 0
            out?.write(disp)
            writeShort(delay)
            out?.write(0)
            out?.write(0)
        }

        private fun writeImageDesc() {
            out?.write(0x2c)
            writeShort(0)
            writeShort(0)
            writeShort(width)
            writeShort(height)
            out?.write(0)
        }

        private fun writePixels() {
            LZWEncoder(width, height, indexedPixels!!, colorDepth).encode(out!!)
        }

        private fun writeShort(value: Int) {
            out?.write(value and 0xff)
            out?.write((value shr 8) and 0xff)
        }

        private fun writeString(s: String) {
            for (c in s) out?.write(c.code)
        }
    }

    private class NeuQuant(
        private val thepicture: ByteArray,
        private val lengthcount: Int,
        private val samplefac: Int,
    ) {
        private val netsize = 256
        private val prime1 = 499
        private val prime2 = 491
        private val prime3 = 487
        private val prime4 = 503
        private val maxnetpos = netsize - 1
        private val netbiasshift = 4
        private val ncycles = 100
        private val intbiasshift = 16
        private val intbias = 1 shl intbiasshift
        private val gammashift = 10
        private val betashift = 10
        private val beta = intbias shr betashift
        private val betagamma = intbias shl (gammashift - betashift)
        private val initrad = netsize shr 3
        private val radiusbiasshift = 6
        private val radiusbias = 1 shl radiusbiasshift
        private val initradius = initrad * radiusbias
        private val radiusdec = 30
        private val alphabiasshift = 10
        private val initalpha = 1 shl alphabiasshift
        private val radbiasshift = 8
        private val radbias = 1 shl radbiasshift
        private val alpharadbshift = alphabiasshift + radbiasshift
        private val alpharadbias = 1 shl alpharadbshift

        private val network = Array(netsize) { IntArray(4) }
        private val netindex = IntArray(256)
        private val bias = IntArray(netsize)
        private val freq = IntArray(netsize)
        private val radpower = IntArray(initrad)

        init {
            for (i in 0 until netsize) {
                val p = network[i]
                p[0] = (i shl (netbiasshift + 8)) / netsize
                p[1] = (i shl (netbiasshift + 8)) / netsize
                p[2] = (i shl (netbiasshift + 8)) / netsize
                freq[i] = intbias / netsize
                bias[i] = 0
            }
        }

        fun process(): ByteArray {
            learn()
            unbiasnet()
            inxbuild()
            return colorMap()
        }

        fun map(b: Int, g: Int, r: Int): Int {
            var bestd = 1000
            var best = -1
            var i = netindex[g]
            var j = i - 1
            while (i < netsize || j >= 0) {
                if (i < netsize) {
                    val p = network[i]
                    var dist = p[1] - g
                    if (dist >= bestd) i = netsize else {
                        i++
                        if (dist < 0) dist = -dist
                        var a = p[0] - b
                        if (a < 0) a = -a
                        dist += a
                        if (dist < bestd) {
                            a = p[2] - r
                            if (a < 0) a = -a
                            dist += a
                            if (dist < bestd) {
                                bestd = dist
                                best = p[3]
                            }
                        }
                    }
                }
                if (j >= 0) {
                    val p = network[j]
                    var dist = g - p[1]
                    if (dist >= bestd) j = -1 else {
                        j--
                        if (dist < 0) dist = -dist
                        var a = p[0] - b
                        if (a < 0) a = -a
                        dist += a
                        if (dist < bestd) {
                            a = p[2] - r
                            if (a < 0) a = -a
                            dist += a
                            if (dist < bestd) {
                                bestd = dist
                                best = p[3]
                            }
                        }
                    }
                }
            }
            return best
        }

        private fun colorMap(): ByteArray {
            val map = ByteArray(3 * netsize)
            val index = IntArray(netsize)
            for (i in 0 until netsize) index[network[i][3]] = i
            var k = 0
            for (i in 0 until netsize) {
                val j = index[i]
                map[k++] = network[j][2].toByte()
                map[k++] = network[j][1].toByte()
                map[k++] = network[j][0].toByte()
            }
            return map
        }

        private fun inxbuild() {
            var previouscol = 0
            var startpos = 0
            for (i in 0 until netsize) {
                val p = network[i]
                var smallpos = i
                var smallval = p[1]
                for (j in i + 1 until netsize) {
                    val q = network[j]
                    if (q[1] < smallval) {
                        smallpos = j
                        smallval = q[1]
                    }
                }
                val q = network[smallpos]
                if (i != smallpos) {
                    var j = q[0]
                    q[0] = p[0]
                    p[0] = j
                    j = q[1]
                    q[1] = p[1]
                    p[1] = j
                    j = q[2]
                    q[2] = p[2]
                    p[2] = j
                    j = q[3]
                    q[3] = p[3]
                    p[3] = j
                }
                if (smallval != previouscol) {
                    netindex[previouscol] = (startpos + i) shr 1
                    for (j in previouscol + 1 until smallval) netindex[j] = i
                    previouscol = smallval
                    startpos = i
                }
            }
            netindex[previouscol] = (startpos + maxnetpos) shr 1
            for (j in previouscol + 1 until 256) netindex[j] = maxnetpos
        }

        private fun learn() {
            val alphadec = 30 + ((samplefac - 1) / 3)
            val samplepixels = lengthcount / (3 * samplefac)
            var delta = samplepixels / ncycles
            var alpha = initalpha
            var radius = initradius

            var rad = radius shr radiusbiasshift
            if (rad <= 1) rad = 0
            for (i in 0 until rad) {
                radpower[i] = alpha * (((rad * rad - i * i) * radbias) / (rad * rad))
            }

            val step =
                when {
                    lengthcount < 500009 -> 3 * prime1
                    lengthcount % prime1 != 0 -> 3 * prime1
                    lengthcount % prime2 != 0 -> 3 * prime2
                    lengthcount % prime3 != 0 -> 3 * prime3
                    else -> 3 * prime4
                }

            var pix = 0
            for (i in 0 until samplepixels) {
                val b = (thepicture[pix].toInt() and 0xff) shl netbiasshift
                val g = (thepicture[pix + 1].toInt() and 0xff) shl netbiasshift
                val r = (thepicture[pix + 2].toInt() and 0xff) shl netbiasshift
                val j = contest(b, g, r)
                altersingle(alpha, j, b, g, r)
                if (rad != 0) alterneigh(rad, j, b, g, r)
                pix += step
                if (pix >= lengthcount) pix -= lengthcount
                if (delta == 0) delta = 1
                if (i % delta == 0) {
                    alpha -= alpha / alphadec
                    radius -= radius / radiusdec
                    rad = radius shr radiusbiasshift
                    if (rad <= 1) rad = 0
                    for (k in 0 until rad) {
                        radpower[k] = alpha * (((rad * rad - k * k) * radbias) / (rad * rad))
                    }
                }
            }
        }

        private fun unbiasnet() {
            for (i in 0 until netsize) {
                network[i][0] = network[i][0] shr netbiasshift
                network[i][1] = network[i][1] shr netbiasshift
                network[i][2] = network[i][2] shr netbiasshift
                network[i][3] = i
            }
        }

        private fun alterneigh(rad: Int, i: Int, b: Int, g: Int, r: Int) {
            var lo = i - rad
            if (lo < -1) lo = -1
            var hi = i + rad
            if (hi > netsize) hi = netsize
            var j = i + 1
            var k = i - 1
            var m = 1
            while (j < hi || k > lo) {
                val a = radpower[m++]
                if (j < hi) {
                    val p = network[j++]
                    p[0] -= (a * (p[0] - b)) / alpharadbias
                    p[1] -= (a * (p[1] - g)) / alpharadbias
                    p[2] -= (a * (p[2] - r)) / alpharadbias
                }
                if (k > lo) {
                    val p = network[k--]
                    p[0] -= (a * (p[0] - b)) / alpharadbias
                    p[1] -= (a * (p[1] - g)) / alpharadbias
                    p[2] -= (a * (p[2] - r)) / alpharadbias
                }
            }
        }

        private fun altersingle(alpha: Int, i: Int, b: Int, g: Int, r: Int) {
            val n = network[i]
            n[0] -= (alpha * (n[0] - b)) / initalpha
            n[1] -= (alpha * (n[1] - g)) / initalpha
            n[2] -= (alpha * (n[2] - r)) / initalpha
        }

        private fun contest(b: Int, g: Int, r: Int): Int {
            var bestd = Int.MAX_VALUE
            var bestbiasd = bestd
            var bestpos = -1
            var bestbiaspos = bestpos
            for (i in 0 until netsize) {
                val n = network[i]
                var dist = kotlin.math.abs(n[0] - b)
                dist += kotlin.math.abs(n[1] - g)
                dist += kotlin.math.abs(n[2] - r)
                if (dist < bestd) {
                    bestd = dist
                    bestpos = i
                }
                val biasdist = dist - (bias[i] shr (intbiasshift - netbiasshift))
                if (biasdist < bestbiasd) {
                    bestbiasd = biasdist
                    bestbiaspos = i
                }
                val betafreq = freq[i] shr betashift
                freq[i] -= betafreq
                bias[i] += betafreq shl gammashift
            }
            freq[bestpos] += beta
            bias[bestpos] -= betagamma
            return bestbiaspos
        }
    }

    private class LZWEncoder(
        private val imgW: Int,
        private val imgH: Int,
        private val pixAry: ByteArray,
        private val initCodeSize: Int,
    ) {
        private val eof = -1
        private var curPixel = 0
        private var codeSize = 0
        private var clearCode = 0
        private var eofCode = 0
        private var freeEnt = 0
        private var maxCode = 0
        private var gInitBits = 0
        private val hSize = 5003
        private val htab = IntArray(hSize)
        private val codetab = IntArray(hSize)
        private var curAccum = 0
        private var curBits = 0
        private val accum = ByteArray(256)
        private var aCount = 0

        fun encode(os: OutputStream) {
            os.write(initCodeSize)
            compress(initCodeSize + 1, os)
            os.write(0)
        }

        private fun compress(initBits: Int, outs: OutputStream) {
            gInitBits = initBits
            codeSize = gInitBits
            clearCode = 1 shl (initBits - 1)
            eofCode = clearCode + 1
            freeEnt = clearCode + 2
            maxCode = (1 shl codeSize) - 1

            for (i in 0 until hSize) htab[i] = -1
            outputCode(clearCode, outs)

            var ent = nextPixel()
            if (ent == eof) {
                outputCode(eofCode, outs)
                flushBits(outs)
                return
            }

            var c: Int
            while (nextPixel().also { c = it } != eof) {
                val fcode = (c shl 12) + ent
                var i = (c shl 4) xor ent

                if (htab[i] == fcode) {
                    ent = codetab[i]
                    continue
                }

                if (htab[i] >= 0) {
                    var disp = hSize - i
                    if (i == 0) disp = 1
                    while (true) {
                        i -= disp
                        if (i < 0) i += hSize
                        if (htab[i] == fcode) {
                            ent = codetab[i]
                            break
                        }
                        if (htab[i] < 0) break
                    }
                    if (htab[i] == fcode) continue
                }

                outputCode(ent, outs)
                ent = c

                if (freeEnt < 4096) {
                    codetab[i] = freeEnt
                    htab[i] = fcode
                    if (freeEnt > maxCode && codeSize < 12) {
                        codeSize++
                        maxCode = (1 shl codeSize) - 1
                    }
                    freeEnt++
                } else {
                    outputCode(clearCode, outs)
                    for (k in 0 until hSize) htab[k] = -1
                    freeEnt = clearCode + 2
                    codeSize = gInitBits
                    maxCode = (1 shl codeSize) - 1
                }
            }

            outputCode(ent, outs)
            outputCode(eofCode, outs)
            flushBits(outs)
        }

        private fun outputCode(code: Int, outs: OutputStream) {
            curAccum = curAccum or (code shl curBits)
            curBits += codeSize
            while (curBits >= 8) {
                addToBlock((curAccum and 0xff).toByte(), outs)
                curAccum = curAccum shr 8
                curBits -= 8
            }
        }

        private fun flushBits(outs: OutputStream) {
            if (curBits > 0) {
                addToBlock((curAccum and 0xff).toByte(), outs)
            }
            flushBlock(outs)
        }

        private fun addToBlock(b: Byte, outs: OutputStream) {
            accum[aCount++] = b
            if (aCount >= 254) {
                flushBlock(outs)
            }
        }

        private fun flushBlock(outs: OutputStream) {
            if (aCount > 0) {
                outs.write(aCount)
                outs.write(accum, 0, aCount)
                aCount = 0
            }
        }

        private fun nextPixel(): Int {
            if (curPixel >= pixAry.size) return eof
            return pixAry[curPixel++].toInt() and 0xff
        }
    }
}