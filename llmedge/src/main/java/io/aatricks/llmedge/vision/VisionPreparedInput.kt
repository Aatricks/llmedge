package io.aatricks.llmedge.vision

import java.io.File

internal sealed interface VisionPreparedInput : AutoCloseable {
    data object PrimedBuffer : VisionPreparedInput {
        override fun close() = Unit
    }

    data class EmbeddingsBuffer(
        val embeddings: VisionEmbeddings,
    ) : VisionPreparedInput {
        override fun close() = Unit
    }

    data class EmbeddingsFile(
        val embedFile: File,
        val metaFile: File,
        val imageFile: File,
        val cleanupOnClose: Boolean = true,
    ) : VisionPreparedInput {
        override fun close() {
            if (!cleanupOnClose) {
                return
            }
            if (metaFile.exists()) metaFile.delete()
            if (embedFile.exists()) embedFile.delete()
            if (imageFile.exists()) imageFile.delete()
        }
    }
}
