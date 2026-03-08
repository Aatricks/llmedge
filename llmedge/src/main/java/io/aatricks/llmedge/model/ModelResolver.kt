package io.aatricks.llmedge.model

import android.content.Context
import io.aatricks.llmedge.core.ProgressEvent
import java.io.File
import java.io.FileNotFoundException

interface ModelResolver {
    suspend fun resolve(
        context: Context,
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)? = null,
    ): File
}

class DefaultModelResolver(private val store: ModelStore = HuggingFaceModelStore()) : ModelResolver {
    override suspend fun resolve(
        context: Context,
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)?,
    ): File =
        when (spec) {
            is ModelSpec.LocalFile -> {
                if (!spec.file.exists()) {
                    throw FileNotFoundException("Model file not found: ${spec.file.absolutePath}")
                }
                spec.file
            }

            is ModelSpec.HuggingFace ->
                store.resolve(context, spec) { downloaded, total ->
                    onProgress?.invoke(
                        ProgressEvent.Downloading(
                            model = spec,
                            downloadedBytes = downloaded,
                            totalBytes = total,
                        ),
                    )
                }
        }
}
