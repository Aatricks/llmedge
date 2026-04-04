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

internal class GifColorQuantizer(
    private val picture: ByteArray,
    private val length: Int,
    private val sampleFactor: Int,
) {
    private val networkSize = 256
    private val prime1 = 499
    private val prime2 = 491
    private val prime3 = 487
    private val prime4 = 503
    private val maxNetworkPosition = networkSize - 1
    private val netBiasShift = 4
    private val cycleCount = 100
    private val intBiasShift = 16
    private val intBias = 1 shl intBiasShift
    private val gammaShift = 10
    private val betaShift = 10
    private val beta = intBias shr betaShift
    private val betaGamma = intBias shl (gammaShift - betaShift)
    private val initialRadiusBase = networkSize shr 3
    private val radiusBiasShift = 6
    private val radiusBias = 1 shl radiusBiasShift
    private val initialRadius = initialRadiusBase * radiusBias
    private val radiusDecay = 30
    private val alphaBiasShift = 10
    private val initialAlpha = 1 shl alphaBiasShift
    private val radBiasShift = 8
    private val radBias = 1 shl radBiasShift
    private val alphaRadBiasShift = alphaBiasShift + radBiasShift
    private val alphaRadBias = 1 shl alphaRadBiasShift

    private val network = Array(networkSize) { IntArray(4) }
    private val networkIndex = IntArray(256)
    private val bias = IntArray(networkSize)
    private val frequency = IntArray(networkSize)
    private val radiusPower = IntArray(initialRadiusBase)

    init {
        for (index in 0 until networkSize) {
            val entry = network[index]
            entry[0] = (index shl (netBiasShift + 8)) / networkSize
            entry[1] = (index shl (netBiasShift + 8)) / networkSize
            entry[2] = (index shl (netBiasShift + 8)) / networkSize
            frequency[index] = intBias / networkSize
            bias[index] = 0
        }
    }

    fun process(): ByteArray {
        learn()
        unbiasNetwork()
        buildIndex()
        return colorMap()
    }

    fun map(b: Int, g: Int, r: Int): Int {
        var bestDistance = 1000
        var best = -1
        var upper = networkIndex[g]
        var lower = upper - 1
        while (upper < networkSize || lower >= 0) {
            if (upper < networkSize) {
                val entry = network[upper]
                var distance = entry[1] - g
                if (distance >= bestDistance) {
                    upper = networkSize
                } else {
                    upper++
                    if (distance < 0) distance = -distance
                    var componentDistance = entry[0] - b
                    if (componentDistance < 0) componentDistance = -componentDistance
                    distance += componentDistance
                    if (distance < bestDistance) {
                        componentDistance = entry[2] - r
                        if (componentDistance < 0) componentDistance = -componentDistance
                        distance += componentDistance
                        if (distance < bestDistance) {
                            bestDistance = distance
                            best = entry[3]
                        }
                    }
                }
            }
            if (lower >= 0) {
                val entry = network[lower]
                var distance = g - entry[1]
                if (distance >= bestDistance) {
                    lower = -1
                } else {
                    lower--
                    if (distance < 0) distance = -distance
                    var componentDistance = entry[0] - b
                    if (componentDistance < 0) componentDistance = -componentDistance
                    distance += componentDistance
                    if (distance < bestDistance) {
                        componentDistance = entry[2] - r
                        if (componentDistance < 0) componentDistance = -componentDistance
                        distance += componentDistance
                        if (distance < bestDistance) {
                            bestDistance = distance
                            best = entry[3]
                        }
                    }
                }
            }
        }
        return best
    }

    private fun colorMap(): ByteArray {
        val map = ByteArray(3 * networkSize)
        val index = IntArray(networkSize)
        for (position in 0 until networkSize) index[network[position][3]] = position
        var offset = 0
        for (position in 0 until networkSize) {
            val sortedIndex = index[position]
            map[offset++] = network[sortedIndex][2].toByte()
            map[offset++] = network[sortedIndex][1].toByte()
            map[offset++] = network[sortedIndex][0].toByte()
        }
        return map
    }

    private fun buildIndex() {
        var previousColor = 0
        var startPosition = 0
        for (index in 0 until networkSize) {
            val current = network[index]
            var smallestPosition = index
            var smallestValue = current[1]
            for (candidate in index + 1 until networkSize) {
                val compared = network[candidate]
                if (compared[1] < smallestValue) {
                    smallestPosition = candidate
                    smallestValue = compared[1]
                }
            }
            val smallest = network[smallestPosition]
            if (index != smallestPosition) {
                var temp = smallest[0]
                smallest[0] = current[0]
                current[0] = temp
                temp = smallest[1]
                smallest[1] = current[1]
                current[1] = temp
                temp = smallest[2]
                smallest[2] = current[2]
                current[2] = temp
                temp = smallest[3]
                smallest[3] = current[3]
                current[3] = temp
            }
            if (smallestValue != previousColor) {
                networkIndex[previousColor] = (startPosition + index) shr 1
                for (fill in previousColor + 1 until smallestValue) networkIndex[fill] = index
                previousColor = smallestValue
                startPosition = index
            }
        }
        networkIndex[previousColor] = (startPosition + maxNetworkPosition) shr 1
        for (fill in previousColor + 1 until 256) networkIndex[fill] = maxNetworkPosition
    }

    private fun learn() {
        val alphaDecay = 30 + ((sampleFactor - 1) / 3)
        val samplePixels = length / (3 * sampleFactor)
        var delta = samplePixels / cycleCount
        var alpha = initialAlpha
        var radius = initialRadius

        var workingRadius = radius shr radiusBiasShift
        if (workingRadius <= 1) workingRadius = 0
        for (index in 0 until workingRadius) {
            radiusPower[index] =
                alpha * (((workingRadius * workingRadius - index * index) * radBias) / (workingRadius * workingRadius))
        }

        val step =
            when {
                length < 500009 -> 3 * prime1
                length % prime1 != 0 -> 3 * prime1
                length % prime2 != 0 -> 3 * prime2
                length % prime3 != 0 -> 3 * prime3
                else -> 3 * prime4
            }

        var pixelIndex = 0
        for (sampleIndex in 0 until samplePixels) {
            val b = (picture[pixelIndex].toInt() and 0xff) shl netBiasShift
            val g = (picture[pixelIndex + 1].toInt() and 0xff) shl netBiasShift
            val r = (picture[pixelIndex + 2].toInt() and 0xff) shl netBiasShift
            val contestIndex = contest(b, g, r)
            alterSingle(alpha, contestIndex, b, g, r)
            if (workingRadius != 0) alterNeighbors(workingRadius, contestIndex, b, g, r)
            pixelIndex += step
            if (pixelIndex >= length) pixelIndex -= length
            if (delta == 0) delta = 1
            if (sampleIndex % delta == 0) {
                alpha -= alpha / alphaDecay
                radius -= radius / radiusDecay
                workingRadius = radius shr radiusBiasShift
                if (workingRadius <= 1) workingRadius = 0
                for (powerIndex in 0 until workingRadius) {
                    radiusPower[powerIndex] =
                        alpha * (((workingRadius * workingRadius - powerIndex * powerIndex) * radBias) / (workingRadius * workingRadius))
                }
            }
        }
    }

    private fun unbiasNetwork() {
        for (index in 0 until networkSize) {
            network[index][0] = network[index][0] shr netBiasShift
            network[index][1] = network[index][1] shr netBiasShift
            network[index][2] = network[index][2] shr netBiasShift
            network[index][3] = index
        }
    }

    private fun alterNeighbors(radius: Int, index: Int, b: Int, g: Int, r: Int) {
        var lower = index - radius
        if (lower < -1) lower = -1
        var upper = index + radius
        if (upper > networkSize) upper = networkSize
        var upperIndex = index + 1
        var lowerIndex = index - 1
        var powerIndex = 1
        while (upperIndex < upper || lowerIndex > lower) {
            val alphaPower = radiusPower[powerIndex++]
            if (upperIndex < upper) {
                val entry = network[upperIndex++]
                entry[0] -= (alphaPower * (entry[0] - b)) / alphaRadBias
                entry[1] -= (alphaPower * (entry[1] - g)) / alphaRadBias
                entry[2] -= (alphaPower * (entry[2] - r)) / alphaRadBias
            }
            if (lowerIndex > lower) {
                val entry = network[lowerIndex--]
                entry[0] -= (alphaPower * (entry[0] - b)) / alphaRadBias
                entry[1] -= (alphaPower * (entry[1] - g)) / alphaRadBias
                entry[2] -= (alphaPower * (entry[2] - r)) / alphaRadBias
            }
        }
    }

    private fun alterSingle(alpha: Int, index: Int, b: Int, g: Int, r: Int) {
        val entry = network[index]
        entry[0] -= (alpha * (entry[0] - b)) / initialAlpha
        entry[1] -= (alpha * (entry[1] - g)) / initialAlpha
        entry[2] -= (alpha * (entry[2] - r)) / initialAlpha
    }

    private fun contest(b: Int, g: Int, r: Int): Int {
        var bestDistance = Int.MAX_VALUE
        var bestBiasDistance = bestDistance
        var bestPosition = -1
        var bestBiasPosition = bestPosition
        for (index in 0 until networkSize) {
            val entry = network[index]
            var distance = kotlin.math.abs(entry[0] - b)
            distance += kotlin.math.abs(entry[1] - g)
            distance += kotlin.math.abs(entry[2] - r)
            if (distance < bestDistance) {
                bestDistance = distance
                bestPosition = index
            }
            val biasedDistance = distance - (bias[index] shr (intBiasShift - netBiasShift))
            if (biasedDistance < bestBiasDistance) {
                bestBiasDistance = biasedDistance
                bestBiasPosition = index
            }
            val betaFrequency = frequency[index] shr betaShift
            frequency[index] -= betaFrequency
            bias[index] += betaFrequency shl gammaShift
        }
        frequency[bestPosition] += beta
        bias[bestPosition] -= betaGamma
        return bestBiasPosition
    }
}
