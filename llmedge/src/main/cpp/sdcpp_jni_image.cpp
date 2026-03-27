#include "sdcpp_jni_shared.h"

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2Img(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint steps, jfloat cfg, jlong seed,
        jboolean jVaeTiling,
        jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent) {
    (void)thiz;
    if (handlePtr == 0) {
        ALOGE("StableDiffusion not initialized");
        return nullptr;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    if (!handle->ctx) {
        throwJavaException(env, "java/lang/IllegalStateException",
                           "StableDiffusion diffusion context is null (T5-only handle). Load a diffusion model (or use *WithPrecomputedCondition) before calling txt2img.");
        return nullptr;
    }
    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    sample.guidance.txt_cfg = cfg > 0 ? cfg : 7.0f;
    sample.flow_shift = handle->flowShift;

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.prompt = prompt;
    gen.negative_prompt = negative;
    gen.width = width;
    gen.height = height;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.batch_count = 1;
    gen.vae_tiling_params.enabled = jVaeTiling ? true : false;
    apply_easycache_compat(&gen.cache,
                           jEasyCacheEnabled == JNI_TRUE,
                           static_cast<float>(jEasyCacheReuseThreshold),
                           static_cast<float>(jEasyCacheStartPercent),
                           static_cast<float>(jEasyCacheEndPercent));

    sd_image_t* out = nullptr;
    try {
        out = generate_image(handle->ctx, &gen);
    } catch (const std::exception& e) {
        if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
        if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }

    if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
    if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);

    if (!out || !out[0].data) {
        ALOGE("generate_image failed");
        return nullptr;
    }

    const size_t byteCount = static_cast<size_t>(out[0].width) * out[0].height * out[0].channel;
    jbyteArray jbytes = env->NewByteArray(static_cast<jsize>(byteCount));
    if (!jbytes) {
        free(out[0].data);
        free(out);
        return nullptr;
    }
    env->SetByteArrayRegion(jbytes, 0, static_cast<jsize>(byteCount), reinterpret_cast<jbyte*>(out[0].data));

    free(out[0].data);
    free(out);

    return jbytes;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2ImgArgb(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint steps, jfloat cfg, jlong seed,
        jboolean jVaeTiling,
        jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent) {
    (void)thiz;
    if (handlePtr == 0) {
        ALOGE("StableDiffusion not initialized");
        return nullptr;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    if (!handle->ctx) {
        throwJavaException(env, "java/lang/IllegalStateException",
                           "StableDiffusion diffusion context is null (T5-only handle).");
        return nullptr;
    }
    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    sample.guidance.txt_cfg = cfg > 0 ? cfg : 7.0f;
    sample.flow_shift = handle->flowShift;

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.prompt = prompt;
    gen.negative_prompt = negative;
    gen.width = width;
    gen.height = height;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.batch_count = 1;
    gen.vae_tiling_params.enabled = jVaeTiling ? true : false;
    apply_easycache_compat(&gen.cache,
                           jEasyCacheEnabled == JNI_TRUE,
                           static_cast<float>(jEasyCacheReuseThreshold),
                           static_cast<float>(jEasyCacheStartPercent),
                           static_cast<float>(jEasyCacheEndPercent));

    sd_image_t* out = nullptr;
    try {
        out = generate_image(handle->ctx, &gen);
    } catch (const std::exception& e) {
        if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
        if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }

    if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
    if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);

    if (!out || !out[0].data) {
        ALOGE("generate_image failed");
        return nullptr;
    }

    jintArray result = rgb_to_argb_int_array(env, out[0].data, out[0].width, out[0].height, out[0].channel);

    free(out[0].data);
    free(out);

    return result;
}
