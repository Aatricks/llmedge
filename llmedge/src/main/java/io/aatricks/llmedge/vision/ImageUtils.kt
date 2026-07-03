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

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.OutputStream

/**
 * Image processing helpers: decoding, scaling, OCR-friendly filtering, EXIF-aware source loading,
 * RGB conversion, and GIF export used by the vision and image-generation flows.
 */
object ImageUtils {
    suspend fun imageToBitmap(context: Context, source: ImageSource): Bitmap =
        ImageBitmapSupport.imageToBitmap(context, source)

    suspend fun imageToFile(
        context: Context,
        source: ImageSource,
        filename: String = "temp_image.jpg",
    ): File = ImageBitmapSupport.imageToFile(context, source, filename)

    /**
     * Orientation-aware preprocessing for file-, URI-, and byte-backed sources.
     *
     * EXIF orientation is applied when metadata is available before scaling/enhancement.
     */
    suspend fun preprocessImage(
        context: Context,
        source: ImageSource,
        maxDimension: Int = ImagePreprocessingSupport.MAX_DIMENSION,
        enhance: Boolean = false,
    ): Bitmap = ImageBitmapSupport.preprocessImageSource(context, source, maxDimension, enhance)

    /** Preprocess a decoded bitmap for OCR or vision use. */
    fun preprocessBitmap(
        bitmap: Bitmap,
        maxDimension: Int = ImagePreprocessingSupport.MAX_DIMENSION,
        enhance: Boolean = false,
    ): Bitmap = ImagePreprocessingSupport.preprocessBitmap(bitmap, maxDimension, enhance)

    /**
     * Deprecated bitmap-only overload retained for source compatibility.
     *
     * Raw [Bitmap] instances do not carry EXIF metadata, so [correctOrientation] cannot be
     * honored here. Use [preprocessBitmap] or the source-aware [preprocessImage] overload instead.
     */
    @Deprecated(
        message = "Bitmap inputs do not retain EXIF orientation; use preprocessBitmap(bitmap, ...) or preprocessImage(context, source, ...).",
        replaceWith = ReplaceWith("preprocessBitmap(bitmap, maxDimension = maxDimension, enhance = enhance)"),
    )
    fun preprocessImage(
        bitmap: Bitmap,
        correctOrientation: Boolean = true,
        maxDimension: Int = ImagePreprocessingSupport.MAX_DIMENSION,
        enhance: Boolean = false,
    ): Bitmap = ImagePreprocessingSupport.preprocessBitmap(bitmap, maxDimension, enhance)

    fun applyExifOrientation(bitmap: Bitmap, imagePath: String): Bitmap =
        ImageBitmapSupport.applyExifOrientation(bitmap, imagePath)

    /**
     * Decode an image file, applying EXIF orientation. Pass [maxDimension] > 0 to subsample
     * large camera captures at decode time (a 50 MP JPEG otherwise decodes to ~200 MB of
     * pixels — over Android's canvas draw limit). 0 keeps the full resolution.
     */
    @JvmOverloads
    fun fileToBitmap(file: File, maxDimension: Int = 0): Bitmap =
        ImageBitmapSupport.fileToBitmap(file, maxDimension)

    suspend fun imageToByteArray(
        context: Context,
        source: ImageSource,
        format: Bitmap.CompressFormat = Bitmap.CompressFormat.JPEG,
        quality: Int = 90,
    ): ByteArray = ImageBitmapSupport.imageToByteArray(context, source, format, quality)

    fun rgbBytesToBitmap(
        rgb: ByteArray,
        width: Int,
        height: Int,
        pixels: IntArray? = null,
    ): Bitmap = ImagePixelSupport.rgbBytesToBitmap(rgb, width, height, pixels)

    fun createAnimatedGif(
        frames: List<Bitmap>,
        delayMs: Int = 100,
        output: OutputStream,
        loop: Int = 0,
        quality: Int = 10,
    ) = ImageGifEncoderSupport.createAnimatedGif(frames, delayMs, output, loop, quality)
}
