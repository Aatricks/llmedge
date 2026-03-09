package io.aatricks.llmedge.runtime

import android.util.Log
import io.aatricks.llmedge.image.diffusion.StableDiffusion

/**
 * Helper for determining optimal flash attention settings based on generation parameters and
 * hardware capabilities.
 */
object FlashAttentionHelper {
    private const val TAG = "FlashAttentionHelper"

    // Cache Vulkan availability to avoid expensive JNI query on every call
    private val hasVulkanSupport: Boolean by lazy {
        try {
            StableDiffusion.getVulkanDeviceCount() > 0
        } catch (_: Exception) {
            false
        }
    }

    /**
     * Determines if flash attention should be enabled for image generation
     *
     * Flash attention is beneficial for:
     * - Long sequences (>512 tokens)
     * - Sequences divisible by 256 (hardware requirement)
     * - Devices with Vulkan support
     *
     * @param width Image width
     * @param height Image height
     * @param steps Number of diffusion steps
     * @param forceEnable Override to always enable
     * @return true if flash attention should be used
     */
    fun shouldUseFlashAttention(
            width: Int,
            height: Int,
            steps: Int = 20,
            forceEnable: Boolean? = null
    ): Boolean {
        // Explicit override
        if (forceEnable != null) return forceEnable

        // Estimate sequence length for Stable Diffusion
        // seq_len ≈ (width/8) * (height/8) for latent space
        val seqLen = estimateSequenceLength(width, height)

        // Flash attention requirements
        val isDivisible = seqLen % 256 == 0
        val isLongEnough = seqLen >= 256

        // Check Vulkan availability (cached)
        val hasVulkan = hasVulkanSupport

        val shouldUse = isDivisible && isLongEnough && hasVulkan

        Log.d(
                TAG,
            "Flash attention decision: width=$width, height=$height, steps=$steps, " +
                        "seqLen=$seqLen, divisible=$isDivisible, long=$isLongEnough, " +
                        "vulkan=$hasVulkan -> $shouldUse"
        )

        return shouldUse
    }

    /** Estimates the sequence length for Stable Diffusion based on image dimensions */
    private fun estimateSequenceLength(width: Int, height: Int): Int {
        // SD processes in latent space (1/8 resolution)
        return (width / 8) * (height / 8)
    }

    /**
     * Determines if flash attention should be enabled for LLM inference.
     *
     * Flash attention is beneficial for LLMs when context sizes are large
     * and Vulkan GPU acceleration is available.
     *
     * @param contextSize The context size (in tokens) for the LLM
     * @return true if flash attention should be used
     */
    fun shouldUseFlashAttentionForLLM(contextSize: Int): Boolean {
        return hasVulkanSupport && contextSize >= 512
    }

    /**
     * Suggests optimal image dimensions for flash attention Returns dimensions that are
     * flash-attention friendly (divisible by 64)
     */
    fun suggestOptimalDimensions(targetWidth: Int, targetHeight: Int): Pair<Int, Int> {
        // Round to nearest 64 (which ensures divisibility by 8 for latent space)
        val optimalWidth = ((targetWidth + 31) / 64) * 64
        val optimalHeight = ((targetHeight + 31) / 64) * 64
        return Pair(optimalWidth, optimalHeight)
    }
}
