package io.aatricks.llmedge.image.diffusion.internal

import android.content.Context
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.StableDiffusionLoadRequest
import io.aatricks.llmedge.image.diffusion.StableDiffusionLoadSupport
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal object StableDiffusionLoader {
    suspend fun load(
        context: Context,
        request: StableDiffusionLoadRequest,
    ): StableDiffusion =
        loadInternal(
            context = context,
            request = request,
        )

    suspend fun loadFromHuggingFace(
        context: Context,
        request: StableDiffusionLoadRequest,
        onProgress: ((name: String, downloaded: Long, total: Long?) -> Unit)?,
    ): StableDiffusion =
        withContext(Dispatchers.IO) {
            val resolved =
                StableDiffusionLoadSupport.resolveWanAssets(
                    context = context,
                    request = request.assets,
                    onProgress = onProgress,
                    validateResolvedAssets = StableDiffusionLoadSupport::validateResolvedAssets,
                    inferVideoModelMetadata = StableDiffusionLoadSupport::inferVideoModelMetadata,
                )

            StableDiffusionLoadSupport.createLoadedInstance(
                context = context,
                resolved = resolved,
                request = request,
            )
        }

    private suspend fun loadInternal(
        context: Context,
        request: StableDiffusionLoadRequest,
    ): StableDiffusion =
        withContext(Dispatchers.IO) {
            val resolved =
                StableDiffusionLoadSupport.resolveRequestedAssets(
                    context = context,
                    request = request.assets,
                    validateResolvedAssets = StableDiffusionLoadSupport::validateResolvedAssets,
                    inferVideoModelMetadata = StableDiffusionLoadSupport::inferVideoModelMetadata,
                    onFallback = StableDiffusionLoadSupport::logLoadFallback,
                )

            StableDiffusionLoadSupport.createLoadedInstance(
                context = context,
                resolved = resolved,
                request = request,
            )
        }
}
