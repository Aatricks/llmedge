package io.aatricks.llmedge.model

import android.content.Context
import io.aatricks.llmedge.core.ProgressEvent
import java.io.File

class ModelManager internal constructor(
    private val context: Context,
    private val resolver: ModelResolver,
) {
    suspend fun resolve(
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)? = null,
    ): File = resolver.resolve(context, spec, onProgress)

    suspend fun prefetch(
        spec: ModelSpec,
        onProgress: ((ProgressEvent.Downloading) -> Unit)? = null,
    ): File = resolve(spec, onProgress)
}
