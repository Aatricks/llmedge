package io.aatricks.llmedge.image

import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.StableDiffusion

internal fun StableDiffusion.resolveEasyCacheParams(
    requested: EasyCacheParams,
    onUnsupported: (() -> Unit)? = null,
): EasyCacheParams {
    if (!requested.enabled) {
        return requested.copy(enabled = false)
    }
    if (isEasyCacheSupported()) {
        return requested
    }
    onUnsupported?.invoke()
    return requested.copy(enabled = false)
}
