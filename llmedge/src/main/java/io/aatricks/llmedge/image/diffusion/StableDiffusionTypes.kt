/*
 * Copyright (C) 2024 LLMEdge Contributors
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

package io.aatricks.llmedge.image.diffusion

import android.graphics.Bitmap

/**
 * Configuration types, parameter classes, and enumerations for Stable Diffusion operations.
 *
 * Extracted from StableDiffusion.kt to separate configuration from orchestration and JNI code.
 */

data class GenerateParams(
        val prompt: String,
        val negative: String = "",
        val width: Int = 512,
        val height: Int = 512,
        val steps: Int = 20,
        val cfgScale: Float = 7.0f,
        val seed: Long = 42L,
        val vaeTiling: Boolean = true,
        val easyCacheParams: EasyCacheParams = EasyCacheParams()
) {
    init {
        require(width > 0) { "Width must be positive" }
        require(height > 0) { "Height must be positive" }
        require(width % 64 == 0) { "Width must be a multiple of 64" }
        require(height % 64 == 0) { "Height must be a multiple of 64" }
    }
}

data class EasyCacheParams(
        val enabled: Boolean = false,
        val reuseThreshold: Float = 0.2f,
        val startPercent: Float = 0.15f,
        val endPercent: Float = 0.95f,
)

/** Sample methods for diffusion models. Maps to native sample_method_t enum values. */
enum class SampleMethod(val id: Int) {
    /** Let native code choose the default for the model type */
    DEFAULT(0),
    /** Euler sampler - Default and recommended for DiT models (Flux/SD3/Wan) */
    EULER(1),
    /** Heun sampler - Higher quality, 2x computation. Works with all models. */
    HEUN(2),
    /** DPM2 sampler - Best for U-Net models (SD1.x/SD2.x/SDXL). Not recommended for Wan. */
    DPM2(3),
    /** DPM++ 2S Ancestral - Best for U-Net models. Not recommended for Wan. */
    DPMPP2S_A(4),
    /** DPM++ 2M - Best for U-Net models. Not recommended for Wan video generation. */
    DPMPP2M(5),
    /** DPM++ 2M v2 - Best for U-Net models. Not recommended for Wan. */
    DPMPP2MV2(6),
    /** IPNDM - Fast sampler */
    IPNDM(7),
    /** IPNDM v */
    IPNDM_V(8),
    /** Latent Consistency Models - Requires LCM-distilled models. NOT compatible with Wan. */
    LCM(9),
    /** DDIM Trailing */
    DDIM_TRAILING(10),
    /** TCD */
    TCD(11),
    /**
     * Euler Ancestral - Default for U-Net models (SD1.x/SD2.x/SDXL). May work with Wan but
     * EULER is preferred.
     */
    EULER_A(12);

    companion object {
        fun fromId(id: Int): SampleMethod = values().firstOrNull { it.id == id } ?: DEFAULT

        /** Samplers recommended for Wan video generation */
        val WAN_RECOMMENDED = listOf(DEFAULT, EULER, HEUN)

        /** Samplers that are NOT compatible with Wan (produce blank/noise output) */
        val WAN_INCOMPATIBLE = listOf(LCM, DPMPP2M, DPMPP2MV2, DPM2, DPMPP2S_A)
    }
}

/** Noise schedulers for diffusion models. Maps to native scheduler_t enum values. */
enum class Scheduler(val id: Int) {
    /** Let native code choose the default scheduler */
    DEFAULT(0),
    /** Discrete scheduler */
    DISCRETE(1),
    /** Karras scheduler - Often better quality */
    KARRAS(2),
    /** Exponential scheduler */
    EXPONENTIAL(3),
    /** AYS scheduler */
    AYS(4),
    /** GITS scheduler */
    GITS(5),
    /** SGM Uniform scheduler */
    SGM_UNIFORM(6),
    /** Simple scheduler */
    SIMPLE(7),
    /** Smoothstep scheduler */
    SMOOTHSTEP(8);

    companion object {
        fun fromId(id: Int): Scheduler = values().firstOrNull { it.id == id } ?: DEFAULT

        /** Schedulers known to work reliably with Wan video generation */
        val WAN_RECOMMENDED = listOf(DEFAULT)
    }
}

data class VideoGenerateParams(
        val prompt: String,
        val negative: String = "",
        val width: Int = 512,
        val height: Int = 512,
        val videoFrames: Int = 16,
        val steps: Int = 20,
        val cfgScale: Float = 7.0f,
        val seed: Long = -1L,
        val initImage: Bitmap? = null,
        val strength: Float = 0.8f,
        val vaceStrength: Float = 1.0f,
        val sampleMethod: SampleMethod = SampleMethod.DEFAULT,
        val scheduler: Scheduler = Scheduler.DEFAULT,
        val easyCacheParams: EasyCacheParams = EasyCacheParams()
) {
    init {
        require(width > 0) { "Width must be positive" }
        require(height > 0) { "Height must be positive" }
        require(width % 64 == 0) { "Width must be a multiple of 64" }
        require(height % 64 == 0) { "Height must be a multiple of 64" }
    }
    /**
     * Calculate the actual number of frames that will be generated. Wan model uses formula:
     * actual_frames = (videoFrames-1)/4*4+1 Examples: 5→5, 8→5, 9→9, 10→9, 12→9, 13→13
     */
    fun actualFrameCount(): Int = (videoFrames - 1) / 4 * 4 + 1

    fun validate(): Result<Unit> = runCatching {
        require(prompt.isNotBlank()) { "Prompt cannot be blank" }
        require(width % 64 == 0 && width in 256..960) {
            "Width must be a multiple of 64 in range 256..960"
        }
        require(height % 64 == 0 && height in 256..960) {
            "Height must be a multiple of 64 in range 256..960"
        }
        // Wan model uses formula: actual_frames = (videoFrames-1)/4*4+1
        // So 1-4 -> 1 frame, 5-8 -> 5 frames, 9-12 -> 9 frames, etc.
        require(videoFrames in 1..64) {
            "Frame count must be between 1 and 64. Note: Wan model rounds to (n-1)/4*4+1, so use 5+ for multiple frames"
        }
        require(steps in 1..50) { "Steps must be between 1 and 50" }
        require(cfgScale in 1.0f..15.0f) { "CFG scale must be between 1.0 and 15.0" }
        require(strength in 0.0f..1.0f) { "Strength must be between 0.0 and 1.0" }
        require(seed >= -1L) { "Seed must be -1 or non-negative" }

        // Validate init image + strength consistency (I2V mode)
        if (initImage != null) {
            require(strength > 0.0f) {
                "When initImage is provided (I2V mode), strength must be > 0.0"
            }
        }
    }

    fun withPrompt(prompt: String): VideoGenerateParams = copy(prompt = prompt)

    companion object {
        fun default(prompt: String = "") = VideoGenerateParams(prompt = prompt)

        /**
         * Get the recommended videoFrames value to generate exactly N frames. Since Wan uses
         * (n-1)/4*4+1, to get exactly N frames you need: 1 frame → 1-4, 5 frames → 5-8, 9
         * frames → 9-12, etc.
         */
        fun recommendedFrameInput(desiredFrames: Int): Int {
            require(desiredFrames >= 1) { "Desired frames must be at least 1" }
            // Reverse the formula: to get N, input N is fine if N = (N-1)/4*4+1
            // Otherwise input N+3 at most
            return if (desiredFrames == 1) 1 else ((desiredFrames - 1) / 4) * 4 + 5
        }
    }
}

enum class LoraApplyMode(val id: Int) {
    AUTO(0),
    IMMEDIATELY(1),
    AT_RUNTIME(2);
    companion object {
        fun fromId(id: Int): LoraApplyMode = values().firstOrNull { it.id == id } ?: AUTO
    }
}

data class GenerationMetrics(
        val totalTimeSeconds: Float,
        val framesPerSecond: Float,
        val timePerStep: Float,
        val peakMemoryUsageMb: Long,
        val vulkanEnabled: Boolean,
        val frameConversionTimeSeconds: Float = 0f,
) {
    var imageRequestMetrics: ImageRequestMetrics? = null
        internal set

    val averageFrameTime: Float
        get() = if (framesPerSecond > 0f) 1f / framesPerSecond else 0f

    val stepsPerSecond: Float
        get() = if (timePerStep > 0f) 1f / timePerStep else 0f

    val throughput: String
        get() = String.format("%.2f fps", framesPerSecond)

    fun toPrettyString(): String =
            """
            Total time: ${String.format("%.2f", totalTimeSeconds)}s
            Throughput: ${String.format("%.2f", framesPerSecond)} fps
            Average time/step: ${String.format("%.3f", timePerStep)}s
            Peak memory: ${peakMemoryUsageMb}MB
            Vulkan: ${if (vulkanEnabled) "enabled" else "disabled"}
            Frame conversion: ${String.format("%.2f", frameConversionTimeSeconds)}s
        """.trimIndent()

    fun withImageRequestMetrics(metrics: ImageRequestMetrics): GenerationMetrics =
        copy().also { it.imageRequestMetrics = metrics }
}

data class ImageRequestMetrics(
    val runtimeAcquireMs: Long,
    val modelLoadMs: Long,
    val generateMs: Long,
    val cacheHit: Boolean,
    val backend: String,
    val flashAttentionEnabled: Boolean,
    val easyCacheEnabled: Boolean,
    val width: Int,
    val height: Int,
    val steps: Int,
) {
    val totalWallTimeMs: Long
        get() = runtimeAcquireMs + generateMs
}

fun interface VideoProgressCallback {
    fun onProgress(
            step: Int,
            totalSteps: Int,
            currentFrame: Int,
            totalFrames: Int,
            timePerStep: Float,
    )
}

/**
 * Container for precomputed text-conditioning arrays returned by the native
 * `sd_precompute_condition` API. Each field is optional and will be null if that tensor is not
 * used for the given model / prompt.
 */
data class PrecomputedCondition(
        val cCrossAttn: FloatArray? = null,
        val cCrossAttnDims: IntArray? = null,
        val cVector: FloatArray? = null,
        val cVectorDims: IntArray? = null,
        val cConcat: FloatArray? = null,
        val cConcatDims: IntArray? = null,
)

internal data class VideoModelMetadata(
        val architecture: String?,
        val modelType: String?,
        val parameterCount: String?,
        val mobileSupported: Boolean,
        val tags: Set<String>,
        val filename: String,
)
