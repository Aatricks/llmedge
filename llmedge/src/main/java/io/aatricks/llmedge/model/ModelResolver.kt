package io.aatricks.llmedge.model

import android.content.Context
import io.aatricks.llmedge.core.ProgressEvent
import java.io.File

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
                ModelFileValidator.requireReadableFile(spec.file, "Model")
            }

            is ModelSpec.HuggingFace ->
                ModelFileValidator.requireReadableFile(
                    store.resolve(context, spec) { downloaded, total ->
                        onProgress?.invoke(
                            ProgressEvent.Downloading(
                                model = spec,
                                downloadedBytes = downloaded,
                                totalBytes = total,
                            ),
                        )
                    },
                    "Model",
                )
        }
}
