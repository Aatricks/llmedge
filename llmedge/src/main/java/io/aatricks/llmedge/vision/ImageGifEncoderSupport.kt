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
}
