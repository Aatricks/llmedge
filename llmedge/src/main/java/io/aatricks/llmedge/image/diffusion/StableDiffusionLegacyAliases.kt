package io.aatricks.llmedge.image.diffusion

@Deprecated("Use SampleMethod.EULER_A instead", ReplaceWith("SampleMethod.EULER_A"))
val StableDiffusion.EULER_A: SampleMethod
    get() = SampleMethod.EULER_A

@Deprecated("Use SampleMethod.DDIM_TRAILING instead", ReplaceWith("SampleMethod.DDIM_TRAILING"))
val StableDiffusion.DDIM: SampleMethod
    get() = SampleMethod.DDIM_TRAILING

@Deprecated("Use SampleMethod.LCM instead", ReplaceWith("SampleMethod.LCM"))
val StableDiffusion.LCM: SampleMethod
    get() = SampleMethod.LCM
