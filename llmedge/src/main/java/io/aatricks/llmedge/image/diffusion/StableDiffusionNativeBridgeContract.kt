package io.aatricks.llmedge.image.diffusion

internal interface StableDiffusionNativeBridgeContract {
    fun txt2img(
        handle: Long,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
        vaeTiling: Boolean,
        easyCacheEnabled: Boolean,
        easyCacheReuseThreshold: Float,
        easyCacheStartPercent: Float,
        easyCacheEndPercent: Float,
    ): ByteArray?

    fun txt2imgArgb(
        handle: Long,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
        vaeTiling: Boolean,
        easyCacheEnabled: Boolean,
        easyCacheReuseThreshold: Float,
        easyCacheStartPercent: Float,
        easyCacheEndPercent: Float,
    ): IntArray? = null

    fun txt2vid(
        handle: Long,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        videoFrames: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
        sampleMethod: SampleMethod,
        scheduler: Scheduler,
        strength: Float,
        initImage: ByteArray?,
        initWidth: Int,
        initHeight: Int,
        vaceStrength: Float,
        easyCacheEnabled: Boolean,
        easyCacheReuseThreshold: Float,
        easyCacheStartPercent: Float,
        easyCacheEndPercent: Float,
    ): Array<ByteArray>?

    fun precomputeCondition(
        handle: Long,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        clipSkip: Int,
    ): PrecomputedCondition? = null

    fun txt2vidWithPrecomputedCondition(
        handle: Long,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        videoFrames: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
        sampleMethod: SampleMethod,
        scheduler: Scheduler,
        strength: Float,
        initImage: ByteArray?,
        initWidth: Int,
        initHeight: Int,
        cond: PrecomputedCondition?,
        uncond: PrecomputedCondition?,
        vaceStrength: Float,
        easyCacheEnabled: Boolean,
        easyCacheReuseThreshold: Float,
        easyCacheStartPercent: Float,
        easyCacheEndPercent: Float,
    ): Array<ByteArray>? =
        txt2vid(
            handle,
            prompt,
            negative,
            width,
            height,
            videoFrames,
            steps,
            cfg,
            seed,
            sampleMethod,
            scheduler,
            strength,
            initImage,
            initWidth,
            initHeight,
            vaceStrength,
            easyCacheEnabled,
            easyCacheReuseThreshold,
            easyCacheStartPercent,
            easyCacheEndPercent,
        )

    fun setProgressCallback(
        handle: Long,
        callback: VideoProgressCallback?,
    )

    fun cancelGeneration(handle: Long)

    fun txt2ImgWithPrecomputedCondition(
        handle: Long,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
        cond: PrecomputedCondition?,
        uncond: PrecomputedCondition?,
        easyCacheEnabled: Boolean,
        easyCacheReuseThreshold: Float,
        easyCacheStartPercent: Float,
        easyCacheEndPercent: Float,
    ): ByteArray? =
        txt2img(
            handle,
            prompt,
            negative,
            width,
            height,
            steps,
            cfg,
            seed,
            true,
            easyCacheEnabled,
            easyCacheReuseThreshold,
            easyCacheStartPercent,
            easyCacheEndPercent,
        )
}
