package io.aatricks.llmedge.image.diffusion

import io.aatricks.llmedge.runtime.ComputeBackend

internal data class StableDiffusionAssetRequest(
    val modelId: String? = null,
    val filename: String? = null,
    val modelPath: String? = null,
    val vaePath: String? = null,
    val t5xxlPath: String? = null,
    val taesdPath: String? = null,
    // FLUX.2 Klein split model: DiT goes here (not modelPath); Qwen3 encoder in llmPath.
    val diffusionModelPath: String? = null,
    val llmPath: String? = null,
    val token: String? = null,
    val forceDownload: Boolean = false,
    val preferSystemDownloader: Boolean = true,
    val loraModelDir: String? = null,
)

internal data class StableDiffusionRuntimeRequest(
    val nThreads: Int,
    val offloadToCpu: Boolean,
    val keepClipOnCpu: Boolean,
    val keepVaeOnCpu: Boolean,
    val flashAttn: Boolean,
    val vaeDecodeOnly: Boolean,
    val sequentialLoad: Boolean?,
    val preferPerformanceMode: Boolean,
    val flowShift: Float,
    val loraApplyMode: LoraApplyMode,
)

internal data class StableDiffusionBackendRequest(
    val allowOpenCl: Boolean,
    val allowVulkan: Boolean,
    val forceVulkan: Boolean,
    val preferredBackend: ComputeBackend? = null,
    val allowBackendFallbackToCpu: Boolean,
)

internal data class StableDiffusionLoadRequest(
    val assets: StableDiffusionAssetRequest,
    val runtime: StableDiffusionRuntimeRequest,
    val backend: StableDiffusionBackendRequest,
)

internal data class StableDiffusionNativeLoadRequest(
    val modelPath: String,
    val vaePath: String?,
    val t5xxlPath: String?,
    val taesdPath: String?,
    val diffusionModelPath: String? = null,
    val llmPath: String? = null,
    val nThreads: Int,
    val enableOpenCl: Boolean,
    val useVulkan: Boolean,
    val offloadToCpu: Boolean,
    val keepClipOnCpu: Boolean,
    val keepVaeOnCpu: Boolean,
    val flashAttn: Boolean,
    val vaeDecodeOnly: Boolean,
    val flowShift: Float,
    val loraModelDir: String?,
    val loraApplyMode: LoraApplyMode,
)
