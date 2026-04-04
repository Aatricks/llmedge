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

import java.io.OutputStream

internal class GifLzwEncoder(
    private val pixels: ByteArray,
    private val initialCodeSize: Int,
) {
    private val eof = -1
    private var currentPixel = 0
    private var codeSize = 0
    private var clearCode = 0
    private var eofCode = 0
    private var freeEntry = 0
    private var maxCode = 0
    private var initialBits = 0
    private val hashSize = 5003
    private val hashTable = IntArray(hashSize)
    private val codeTable = IntArray(hashSize)
    private var currentAccumulator = 0
    private var currentBits = 0
    private val blockBuffer = ByteArray(256)
    private var blockCount = 0

    fun encode(output: OutputStream) {
        output.write(initialCodeSize)
        compress(initialCodeSize + 1, output)
        output.write(0)
    }

    private fun compress(initBits: Int, output: OutputStream) {
        initialBits = initBits
        codeSize = initialBits
        clearCode = 1 shl (initBits - 1)
        eofCode = clearCode + 1
        freeEntry = clearCode + 2
        maxCode = (1 shl codeSize) - 1

        for (index in 0 until hashSize) hashTable[index] = -1
        outputCode(clearCode, output)

        var entry = nextPixel()
        if (entry == eof) {
            outputCode(eofCode, output)
            flushBits(output)
            return
        }

        var code: Int
        while (nextPixel().also { code = it } != eof) {
            val fullCode = (code shl 12) + entry
            var hashIndex = (code shl 4) xor entry

            if (hashTable[hashIndex] == fullCode) {
                entry = codeTable[hashIndex]
                continue
            }

            if (hashTable[hashIndex] >= 0) {
                var displacement = hashSize - hashIndex
                if (hashIndex == 0) displacement = 1
                while (true) {
                    hashIndex -= displacement
                    if (hashIndex < 0) hashIndex += hashSize
                    if (hashTable[hashIndex] == fullCode) {
                        entry = codeTable[hashIndex]
                        break
                    }
                    if (hashTable[hashIndex] < 0) break
                }
                if (hashTable[hashIndex] == fullCode) continue
            }

            outputCode(entry, output)
            entry = code

            if (freeEntry < 4096) {
                codeTable[hashIndex] = freeEntry
                hashTable[hashIndex] = fullCode
                if (freeEntry > maxCode && codeSize < 12) {
                    codeSize++
                    maxCode = (1 shl codeSize) - 1
                }
                freeEntry++
            } else {
                outputCode(clearCode, output)
                for (index in 0 until hashSize) hashTable[index] = -1
                freeEntry = clearCode + 2
                codeSize = initialBits
                maxCode = (1 shl codeSize) - 1
            }
        }

        outputCode(entry, output)
        outputCode(eofCode, output)
        flushBits(output)
    }

    private fun outputCode(code: Int, output: OutputStream) {
        currentAccumulator = currentAccumulator or (code shl currentBits)
        currentBits += codeSize
        while (currentBits >= 8) {
            addToBlock((currentAccumulator and 0xff).toByte(), output)
            currentAccumulator = currentAccumulator shr 8
            currentBits -= 8
        }
    }

    private fun flushBits(output: OutputStream) {
        if (currentBits > 0) {
            addToBlock((currentAccumulator and 0xff).toByte(), output)
        }
        flushBlock(output)
    }

    private fun addToBlock(value: Byte, output: OutputStream) {
        blockBuffer[blockCount++] = value
        if (blockCount >= 254) {
            flushBlock(output)
        }
    }

    private fun flushBlock(output: OutputStream) {
        if (blockCount > 0) {
            output.write(blockCount)
            output.write(blockBuffer, 0, blockCount)
            blockCount = 0
        }
    }

    private fun nextPixel(): Int {
        if (currentPixel >= pixels.size) return eof
        return pixels[currentPixel++].toInt() and 0xff
    }
}
