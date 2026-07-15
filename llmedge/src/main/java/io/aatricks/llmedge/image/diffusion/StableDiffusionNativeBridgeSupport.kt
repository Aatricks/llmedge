package io.aatricks.llmedge.image.diffusion

internal object StableDiffusionNativeBridgeSupport {
    fun defaultProvider(): (StableDiffusion) -> StableDiffusion.NativeBridge = { instance ->
        object : StableDiffusion.NativeBridge {
            override fun txt2img(
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
            ): ByteArray? =
                instance.nativeTxt2Img(
                    handle,
                    prompt,
                    negative,
                    width,
                    height,
                    steps,
                    cfg,
                    seed,
                    vaeTiling,
                    easyCacheEnabled,
                    easyCacheReuseThreshold,
                    easyCacheStartPercent,
                    easyCacheEndPercent,
                )

            override fun txt2imgArgb(
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
            ): IntArray? =
                instance.nativeTxt2ImgArgb(
                    handle,
                    prompt,
                    negative,
                    width,
                    height,
                    steps,
                    cfg,
                    seed,
                    vaeTiling,
                    easyCacheEnabled,
                    easyCacheReuseThreshold,
                    easyCacheStartPercent,
                    easyCacheEndPercent,
                )

            override fun txt2vid(
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
            ): Array<ByteArray>? =
                instance.nativeTxt2Vid(
                    handle,
                    prompt,
                    negative,
                    width,
                    height,
                    videoFrames,
                    steps,
                    cfg,
                    seed,
                    sampleMethod.id,
                    scheduler.id,
                    strength,
                    initImage = initImage,
                    initWidth = initWidth,
                    initHeight = initHeight,
                    vaceStrength = vaceStrength,
                    easyCacheEnabled = easyCacheEnabled,
                    easyCacheReuseThreshold = easyCacheReuseThreshold,
                    easyCacheStartPercent = easyCacheStartPercent,
                    easyCacheEndPercent = easyCacheEndPercent,
                )

            override fun precomputeCondition(
                handle: Long,
                prompt: String,
                negative: String,
                width: Int,
                height: Int,
                clipSkip: Int,
            ): PrecomputedCondition? =
                precomputeCondition(
                    handle = handle,
                    prompt = prompt,
                    negative = negative,
                    width = width,
                    height = height,
                    clipSkip = clipSkip,
                    isVideo = false,
                )

            override fun precomputeCondition(
                handle: Long,
                prompt: String,
                negative: String,
                width: Int,
                height: Int,
                clipSkip: Int,
                isVideo: Boolean,
            ): PrecomputedCondition? =
                instance.nativePrecomputeCondition(handle, prompt, negative, width, height, clipSkip, isVideo)
                    ?.let(StableDiffusionConditionInterop::fromNativeRaw)

            override fun txt2vidWithPrecomputedCondition(
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
                instance.nativeTxt2VidWithPrecomputedCondition(
                    handle,
                    prompt,
                    negative,
                    width,
                    height,
                    videoFrames,
                    steps,
                    cfg,
                    seed,
                    sampleMethod.id,
                    scheduler.id,
                    strength,
                    initImage = initImage,
                    initWidth = initWidth,
                    initHeight = initHeight,
                    cond = StableDiffusionConditionInterop.toNativeArray(cond),
                    uncond = StableDiffusionConditionInterop.toNativeArray(uncond),
                    vaceStrength = vaceStrength,
                    easyCacheEnabled = easyCacheEnabled,
                    easyCacheReuseThreshold = easyCacheReuseThreshold,
                    easyCacheStartPercent = easyCacheStartPercent,
                    easyCacheEndPercent = easyCacheEndPercent,
                )

            override fun setProgressCallback(
                handle: Long,
                callback: VideoProgressCallback?,
            ) {
                instance.nativeSetProgressCallback(handle, callback)
            }

            override fun cancelGeneration(handle: Long) {
                instance.nativeCancelGeneration(handle)
            }

            override fun txt2ImgWithPrecomputedCondition(
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
                txt2ImgWithPrecomputedCondition(
                    handle = handle,
                    prompt = prompt,
                    negative = negative,
                    width = width,
                    height = height,
                    steps = steps,
                    cfg = cfg,
                    seed = seed,
                    vaeTiling = true,
                    cond = cond,
                    uncond = uncond,
                    easyCacheEnabled = easyCacheEnabled,
                    easyCacheReuseThreshold = easyCacheReuseThreshold,
                    easyCacheStartPercent = easyCacheStartPercent,
                    easyCacheEndPercent = easyCacheEndPercent,
                )

            override fun txt2ImgWithPrecomputedCondition(
                handle: Long,
                prompt: String,
                negative: String,
                width: Int,
                height: Int,
                steps: Int,
                cfg: Float,
                seed: Long,
                vaeTiling: Boolean,
                cond: PrecomputedCondition?,
                uncond: PrecomputedCondition?,
                easyCacheEnabled: Boolean,
                easyCacheReuseThreshold: Float,
                easyCacheStartPercent: Float,
                easyCacheEndPercent: Float,
            ): ByteArray? =
                instance.nativeTxt2ImgWithPrecomputedCondition(
                    handle,
                    prompt,
                    negative,
                    width,
                    height,
                    steps,
                    cfg,
                    seed,
                    vaeTiling,
                    StableDiffusionConditionInterop.toNativeArray(cond),
                    StableDiffusionConditionInterop.toNativeArray(uncond),
                    easyCacheEnabled,
                    easyCacheReuseThreshold,
                    easyCacheStartPercent,
                    easyCacheEndPercent,
                )

            override fun upscale(
                esrganPath: String,
                nThreads: Int,
                tileSize: Int,
                backend: String,
                pixels: IntArray,
                width: Int,
                height: Int,
                factor: Int,
                outDims: IntArray,
            ): IntArray? =
                StableDiffusion.nativeUpscale(
                    esrganPath = esrganPath,
                    nThreads = nThreads,
                    tileSize = tileSize,
                    backend = backend,
                    pixels = pixels,
                    width = width,
                    height = height,
                    factor = factor,
                    outDims = outDims
                )
        }
    }
}
