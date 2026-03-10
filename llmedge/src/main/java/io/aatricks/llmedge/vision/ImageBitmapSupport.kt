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
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal object ImageBitmapSupport {
    private const val JPEG_QUALITY = 90

    suspend fun imageToBitmap(context: Context, source: ImageSource): Bitmap = safeWithIO {
        val bitmap =
            when (source) {
                is ImageSource.BitmapSource -> source.bitmap
                is ImageSource.FileSource -> decodeBitmapWithExif(source.file)
                is ImageSource.UriSource -> decodeBitmapFromUri(context, source)
                is ImageSource.ByteArraySource -> decodeBitmapFromBytes(source.bytes)
            }

        bitmap ?: throw IOException("Failed to decode image from source: ${describe(source)}")
    }

    suspend fun imageToFile(
        context: Context,
        source: ImageSource,
        filename: String,
    ): File = safeWithIO {
        val tempFile = File(context.cacheDir, filename)
        when (source) {
            is ImageSource.FileSource -> source.file
            is ImageSource.BitmapSource -> {
                saveBitmapToFile(source.bitmap, tempFile)
                tempFile
            }
            is ImageSource.UriSource -> {
                context.contentResolver.openInputStream(source.uri)?.use { input ->
                    tempFile.outputStream().use { output -> input.copyTo(output) }
                } ?: throw IOException("Failed to open URI: ${source.uri}")
                tempFile
            }
            is ImageSource.ByteArraySource -> {
                tempFile.writeBytes(source.bytes)
                tempFile
            }
        }
    }

    suspend fun imageToByteArray(
        context: Context,
        source: ImageSource,
        format: Bitmap.CompressFormat,
        quality: Int,
    ): ByteArray = safeWithIO {
        when (source) {
            is ImageSource.ByteArraySource -> source.bytes
            is ImageSource.FileSource -> source.file.readBytes()
            is ImageSource.UriSource -> {
                context.contentResolver.openInputStream(source.uri)?.use { stream ->
                    stream.readBytes()
                } ?: throw IOException("Failed to open URI: ${source.uri}")
            }
            is ImageSource.BitmapSource -> {
                ByteArrayOutputStream().use { stream ->
                    source.bitmap.compress(format, quality, stream)
                    stream.toByteArray()
                }
            }
        }
    }

    fun applyExifOrientation(bitmap: Bitmap, imagePath: String): Bitmap =
        try {
            val exif = ExifInterface(imagePath)
            applyOrientation(bitmap, exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL))
        } catch (_: Exception) {
            bitmap
        }

    fun fileToBitmap(file: File): Bitmap =
        decodeBitmapWithExif(file)
            ?: throw IOException("Failed to decode image file: ${file.absolutePath}")

    fun preprocessImageSource(
        context: Context,
        source: ImageSource,
        maxDimension: Int,
        enhance: Boolean,
    ): Bitmap {
        val bitmap =
            when (source) {
                is ImageSource.BitmapSource -> source.bitmap
                is ImageSource.FileSource -> decodeBitmapWithExif(source.file)
                is ImageSource.UriSource -> decodeBitmapFromUriBlocking(context, source)
                is ImageSource.ByteArraySource -> decodeBitmapFromBytes(source.bytes)
            } ?: throw IOException("Failed to decode image from source: ${describe(source)}")

        return ImagePreprocessingSupport.preprocessBitmap(bitmap, maxDimension, enhance)
    }

    private suspend fun <T> safeWithIO(block: suspend () -> T): T =
        try {
            withContext(Dispatchers.IO) { block() }
        } catch (e: Throwable) {
            if (e is NullPointerException) {
                block()
            } else {
                throw e
            }
        }

    private fun decodeBitmapWithExif(file: File): Bitmap? {
        val bitmap = BitmapFactory.decodeFile(file.absolutePath) ?: return null
        return applyExifOrientation(bitmap, file.absolutePath)
    }

    private fun decodeBitmapFromUri(context: Context, source: ImageSource.UriSource): Bitmap? {
        return decodeBitmapFromUriBlocking(context, source)
    }

    private fun decodeBitmapFromUriBlocking(context: Context, source: ImageSource.UriSource): Bitmap? {
        val bytes =
            context.contentResolver.openInputStream(source.uri)?.use { stream ->
                stream.readBytes()
            } ?: return null
        return decodeBitmapFromBytes(bytes)
    }

    private fun decodeBitmapFromBytes(bytes: ByteArray): Bitmap? {
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
        return applyExifOrientation(bitmap, bytes)
    }

    private fun applyExifOrientation(bitmap: Bitmap, bytes: ByteArray): Bitmap =
        try {
            val exif = ExifInterface(ByteArrayInputStream(bytes))
            applyOrientation(bitmap, exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL))
        } catch (_: Exception) {
            bitmap
        }

    private fun applyOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun saveBitmapToFile(bitmap: Bitmap, file: File, quality: Int = JPEG_QUALITY) {
        file.outputStream().use { stream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)
        }
    }

    private fun describe(source: ImageSource): String =
        when (source) {
            is ImageSource.BitmapSource -> "BitmapSource"
            is ImageSource.FileSource -> "File(${source.file.absolutePath})"
            is ImageSource.UriSource -> "Uri(${source.uri})"
            is ImageSource.ByteArraySource -> "ByteArray(len=${source.bytes.size})"
        }
}