package io.aatricks.llmedge.vision

import android.content.Context
import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.io.File

internal class VisionInputPreparer(
    private val context: Context,
    private val jpegQuality: Int,
) {
    suspend fun prepare(
        request: VisionRequest,
        runtime: ManagedVisionRuntime,
        onStatus: ((String) -> Unit)?,
        logStage: (String, String, Long) -> Unit,
    ): VisionPreparedInput {
        val smol = runtime.smol
        val projector = runtime.projector

        smol.clearKvCache()
        onStatus?.invoke("Preparing image")
        val imagePrepStartedNs = System.nanoTime()
        val scaled =
            ImageUtils.preprocessBitmap(
                request.image,
                maxDimension = 672,
                enhance = false,
            )
        logStage("analyze", "image_prep", imagePrepStartedNs)

        return prepareBufferInput(scaled, smol, projector, onStatus, logStage)
            ?: prepareFileInput(scaled, projector, onStatus, logStage)
    }

    private fun prepareBufferInput(
        scaled: Bitmap,
        smol: io.aatricks.llmedge.text.runtime.SmolLM,
        projector: Projector,
        onStatus: ((String) -> Unit)?,
        logStage: (String, String, Long) -> Unit,
    ): VisionPreparedInput? {
        return try {
            val jpegBytes = scaled.toJpegBytes(jpegQuality)
            onStatus?.invoke("Preparing multimodal embeddings")
            val embeddingStartedNs = System.nanoTime()
            val primed = smol.primeImageBuffer(projector.nativeHandle(), jpegBytes, nBatch = 1)
            logStage("analyze", "prime_image_buffer", embeddingStartedNs)
            if (primed) {
                VisionPreparedInput.PrimedBuffer
            } else {
                val bufferStartedNs = System.nanoTime()
                val embeddings = projector.encodeImageBuffer(jpegBytes)
                logStage("analyze", "encode_image_buffer", bufferStartedNs)
                embeddings?.let(VisionPreparedInput::EmbeddingsBuffer)
            }
        } catch (_: UnsatisfiedLinkError) {
            null
        }
    }

    private fun prepareFileInput(
        scaled: Bitmap,
        projector: Projector,
        onStatus: ((String) -> Unit)?,
        logStage: (String, String, Long) -> Unit,
    ): VisionPreparedInput.EmbeddingsFile {
        val imageFile = File.createTempFile("vision_input", ".jpg", context.cacheDir)
        val embedFile = File.createTempFile("vision_prepared", ".bin", context.cacheDir)
        val metaFile = File(embedFile.absolutePath + ".meta.json")

        imageFile.outputStream().use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, jpegQuality, out)
        }

        onStatus?.invoke("Preparing multimodal embeddings")
        val embeddingStartedNs = System.nanoTime()
        val encoded = projector.encodeImageToFile(imageFile.absolutePath, embedFile.absolutePath)
        logStage("analyze", "encode_image_file", embeddingStartedNs)
        check(encoded && metaFile.exists()) {
            "Native projector support is unavailable or failed to produce embeddings."
        }

        return VisionPreparedInput.EmbeddingsFile(
            embedFile = embedFile,
            metaFile = metaFile,
            imageFile = imageFile,
        )
    }

    private fun Bitmap.toJpegBytes(jpegQuality: Int): ByteArray {
        val stream = ByteArrayOutputStream()
        compress(Bitmap.CompressFormat.JPEG, jpegQuality, stream)
        return stream.toByteArray()
    }
}
