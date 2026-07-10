#include "sdcpp_jni_shared.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "jni_utils.h"

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2Vid(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint videoFrames, jint steps, jfloat cfg, jlong seed,
        jint jSampleMethod, jint jScheduler, jfloat jStrength,
        jbyteArray jInitImage, jint initWidth, jint initHeight,
        jfloat jVaceStrength,
        jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent) {
    (void)thiz;
    if (handlePtr == 0) {
        throwJavaException(env, "java/lang/IllegalStateException", "StableDiffusion not initialized");
        return nullptr;
    }
    if (width <= 0 || height <= 0 || videoFrames <= 0) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Invalid video dimensions or frame count");
        return nullptr;
    }

    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    if (!handle->ctx) {
        throwJavaException(env, "java/lang/IllegalStateException",
                           "StableDiffusion diffusion context is null (T5-only handle). Load a diffusion model (or use *WithPrecomputedCondition) before calling txt2vid.");
        return nullptr;
    }
    handle->cancellationRequested.store(false);
    handle->totalFrames = std::max(1, static_cast<int>(videoFrames));
    handle->currentFrame = 0;

    // Standard UTF-8 (not GetStringUTFChars' Modified UTF-8) so emoji and other
    // supplementary characters reach the native tokenizer as valid bytes.
    const std::string promptUtf8 = llmedge_jstring_to_utf8(env, jPrompt);
    const std::string negativeUtf8 = llmedge_jstring_to_utf8(env, jNegative);
    const char* prompt = promptUtf8.c_str();
    const char* negative = negativeUtf8.c_str();

    auto releaseStrings = [&]() {
    };

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    if (cfg > 0.f) sample.guidance.txt_cfg = cfg;
    sample.flow_shift = handle->flowShift;

    sd_vid_gen_params_t gen{};
    sd_vid_gen_params_init(&gen);
    gen.prompt = prompt;
    gen.negative_prompt = negative;
    gen.width = width;
    gen.height = height;
    gen.video_frames = videoFrames;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.vace_strength = jVaceStrength;

    {
        enum sample_method_t mapped_method;
        if (map_sample_method_from_kotlin_id(static_cast<int>(jSampleMethod), &mapped_method)) {
            gen.sample_params.sample_method = mapped_method;
        } else {
            gen.sample_params.sample_method = SAMPLE_METHOD_COUNT;
        }

        enum scheduler_t mapped_sched;
        if (map_scheduler_from_kotlin_id(static_cast<int>(jScheduler), &mapped_sched)) {
            gen.sample_params.scheduler = mapped_sched;
        } else {
            gen.sample_params.scheduler = SCHEDULER_COUNT;
        }
    }

    gen.strength = jStrength;

    std::vector<uint8_t> initImageData;
    if (jInitImage != nullptr) {
        jsize initSize = env->GetArrayLength(jInitImage);
        if (initSize > 0) {
            initImageData.resize(static_cast<size_t>(initSize));
            env->GetByteArrayRegion(jInitImage, 0, initSize, reinterpret_cast<jbyte*>(initImageData.data()));
            if (env->ExceptionCheck()) {
                releaseStrings();
                return nullptr;
            }
            gen.init_image.width = initWidth;
            gen.init_image.height = initHeight;
            gen.init_image.channel = 3;
            gen.init_image.data = initImageData.data();
        }
    }

    apply_easycache_compat(&gen.cache,
                           jEasyCacheEnabled == JNI_TRUE,
                           static_cast<float>(jEasyCacheReuseThreshold),
                           static_cast<float>(jEasyCacheStartPercent),
                           static_cast<float>(jEasyCacheEndPercent));

    handle->stepsPerFrame = 0;
    handle->totalSteps = sample.sample_steps > 0 ? sample.sample_steps : 0;

    SdProgressCallbackGuard callbackGuard(handle);

    sd_image_t* frames = nullptr;
    int numFrames = 0;
    try {
        bool ok = generate_video(handle->ctx, &gen, &frames, &numFrames, nullptr);
        if (!ok) {
            frames = nullptr;
            numFrames = 0;
        }
    } catch (const std::exception& e) {
        releaseStrings();
        const char* clazz = handle->cancellationRequested.load()
                ? "java/util/concurrent/CancellationException"
                : "java/lang/RuntimeException";
        throwJavaException(env, clazz, e.what());
        return nullptr;
    }

    releaseStrings();

    if (!frames || numFrames <= 0) {
        if (frames) {
            free_sd_generated_frames(frames, numFrames);
        }
        throwJavaException(env, "java/lang/IllegalStateException", "Video generation failed");
        return nullptr;
    }

    jobjectArray result = convert_sd_frames_to_java(env, frames, numFrames);
    handle->cancellationRequested.store(false);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeSetProgressCallback(
        JNIEnv* env, jobject, jlong handlePtr, jobject progressCallback) {
    if (handlePtr == 0) {
        throwJavaException(env, "java/lang/IllegalStateException", "StableDiffusion not initialized");
        return;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    if (!handle->jvm && env) {
        env->GetJavaVM(&handle->jvm);
        jni_thread_cache_init(handle->jvm);
    }

    if (!progressCallback) {
        clearProgressCallback(env, handle);
        sd_set_progress_callback(nullptr, nullptr);
        return;
    }

    clearProgressCallback(env, handle);

    jmethodID methodId =
            llmedge_get_callback_method(
                    env,
                    progressCallback,
                    "onProgress",
                    "(IIIIF)V",
                    "java/lang/NoSuchMethodError",
                    "onProgress method not found");
    if (!methodId) {
        return;
    }
    jobject globalRef =
            llmedge_new_global_ref_or_throw(
                    env,
                    progressCallback,
                    "Unable to hold progress callback reference");
    if (!globalRef) {
        return;
    }

    handle->progressCallbackGlobalRef = globalRef;
    handle->progressMethodID = methodId;
    handle->cancellationRequested.store(false);
    sd_set_progress_callback(sd_video_progress_wrapper, handle);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCancelGeneration(
        JNIEnv* env, jobject, jlong handlePtr) {
    (void)env;
    if (handlePtr == 0) {
        return;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    handle->cancellationRequested.store(true);
}
