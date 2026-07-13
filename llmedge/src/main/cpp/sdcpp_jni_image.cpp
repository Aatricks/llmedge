#include "jni_utils.h"
#include "sdcpp_jni_shared.h"

#include <cstddef>
#include <chrono>
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
    handle->cancellationRequested.store(false);
    SdProgressCallbackGuard callbackGuard(handle);
    // Standard UTF-8 (not GetStringUTFChars' Modified UTF-8) so emoji and other
    // supplementary characters reach the native tokenizer as valid bytes.
    const std::string promptUtf8 = llmedge_jstring_to_utf8(env, jPrompt);
    const std::string negativeUtf8 = llmedge_jstring_to_utf8(env, jNegative);
    const char* prompt = promptUtf8.c_str();
    const char* negative = negativeUtf8.c_str();
    SdResolvedPromptLoras resolved = resolve_prompt_loras(prompt, negative, handle->loraModelDir);

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    sample.guidance.txt_cfg = cfg > 0 ? cfg : 7.0f;
    sample.flow_shift = handle->flowShift;

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.prompt = resolved.prompt.c_str();
    gen.negative_prompt = resolved.negativePrompt.c_str();
    gen.width = width;
    gen.height = height;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.batch_count = 1;
    gen.loras = resolved.loras.empty() ? nullptr : resolved.loras.data();
    gen.lora_count = static_cast<uint32_t>(resolved.loras.size());
    gen.vae_tiling_params.enabled = jVaeTiling ? true : false;
    apply_easycache_compat(&gen.cache,
                           jEasyCacheEnabled == JNI_TRUE,
                           static_cast<float>(jEasyCacheReuseThreshold),
                           static_cast<float>(jEasyCacheStartPercent),
                           static_cast<float>(jEasyCacheEndPercent));

    sd_image_t* out = nullptr;
    int numImages = 0;
    bool generated = false;
    const auto t0 = std::chrono::steady_clock::now();
    ALOGI("nativeTxt2Img: generate_image start width=%d height=%d steps=%d promptChars=%zu loraCount=%u", width, height, steps, resolved.prompt.size(), gen.lora_count);
    try {
        generated = generate_image(handle->ctx, &gen, &out, &numImages);
    } catch (const std::exception& e) {
        const char* clazz = handle->cancellationRequested.load()
                ? "java/util/concurrent/CancellationException"
                : "java/lang/RuntimeException";
        throwJavaException(env, clazz, e.what());
        return nullptr;
    }

    if (!generated || numImages < 1 || !out || !out[0].data) {
        ALOGE("generate_image failed");
        if (out) free_sd_images(out, numImages);
        return nullptr;
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0
    ).count();
    ALOGI("nativeTxt2Img: generate_image completed in %lldms", static_cast<long long>(elapsed_ms));

    const size_t byteCount = static_cast<size_t>(out[0].width) * out[0].height * out[0].channel;
    jbyteArray jbytes = env->NewByteArray(static_cast<jsize>(byteCount));
    if (!jbytes) {
        free_sd_images(out, numImages);
        return nullptr;
    }
    env->SetByteArrayRegion(jbytes, 0, static_cast<jsize>(byteCount), reinterpret_cast<jbyte*>(out[0].data));

    free_sd_images(out, numImages);

    handle->cancellationRequested.store(false);
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
    handle->cancellationRequested.store(false);
    SdProgressCallbackGuard callbackGuard(handle);
    // Standard UTF-8 (not GetStringUTFChars' Modified UTF-8) so emoji and other
    // supplementary characters reach the native tokenizer as valid bytes.
    const std::string promptUtf8 = llmedge_jstring_to_utf8(env, jPrompt);
    const std::string negativeUtf8 = llmedge_jstring_to_utf8(env, jNegative);
    const char* prompt = promptUtf8.c_str();
    const char* negative = negativeUtf8.c_str();
    SdResolvedPromptLoras resolved = resolve_prompt_loras(prompt, negative, handle->loraModelDir);

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    sample.guidance.txt_cfg = cfg > 0 ? cfg : 7.0f;
    sample.flow_shift = handle->flowShift;

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.prompt = resolved.prompt.c_str();
    gen.negative_prompt = resolved.negativePrompt.c_str();
    gen.width = width;
    gen.height = height;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.batch_count = 1;
    gen.loras = resolved.loras.empty() ? nullptr : resolved.loras.data();
    gen.lora_count = static_cast<uint32_t>(resolved.loras.size());
    gen.vae_tiling_params.enabled = jVaeTiling ? true : false;
    apply_easycache_compat(&gen.cache,
                           jEasyCacheEnabled == JNI_TRUE,
                           static_cast<float>(jEasyCacheReuseThreshold),
                           static_cast<float>(jEasyCacheStartPercent),
                           static_cast<float>(jEasyCacheEndPercent));

    sd_image_t* out = nullptr;
    int numImages = 0;
    bool generated = false;
    const auto t0 = std::chrono::steady_clock::now();
    ALOGI("nativeTxt2ImgArgb: generate_image start width=%d height=%d steps=%d promptChars=%zu loraCount=%u", width, height, steps, resolved.prompt.size(), gen.lora_count);
    try {
        generated = generate_image(handle->ctx, &gen, &out, &numImages);
    } catch (const std::exception& e) {
        const char* clazz = handle->cancellationRequested.load()
                ? "java/util/concurrent/CancellationException"
                : "java/lang/RuntimeException";
        throwJavaException(env, clazz, e.what());
        return nullptr;
    }

    if (!generated || numImages < 1 || !out || !out[0].data) {
        ALOGE("generate_image failed");
        if (out) free_sd_images(out, numImages);
        return nullptr;
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0
    ).count();
    ALOGI("nativeTxt2ImgArgb: generate_image completed in %lldms", static_cast<long long>(elapsed_ms));

    jintArray result = rgb_to_argb_int_array(env, out[0].data, out[0].width, out[0].height, out[0].channel);

    free_sd_images(out, numImages);

    handle->cancellationRequested.store(false);
    return result;
}
