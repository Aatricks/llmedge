#include "jni_utils.h"
#include "sdcpp_jni_shared.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

// conditioner.hpp drags in the full header-only sd.cpp model graph code, which the
// host-native test harness cannot link (it stubs libstable-diffusion instead).
#ifndef SD_JNI_TESTING
#include "conditioner.hpp"
#endif
#include "ggml-backend.h"
#include "model.h"

static sd_condition_raw_t* reconstruct_condition(JNIEnv* env, jobjectArray condArr) {
    if (!condArr) {
        return nullptr;
    }
    if (env->GetArrayLength(condArr) < 6) {
        return nullptr;
    }

    auto* cond = static_cast<sd_condition_raw_t*>(calloc(1, sizeof(sd_condition_raw_t)));
    if (!cond) {
        throwJavaException(env, "java/lang/OutOfMemoryError", "Failed to allocate memory for condition");
        return nullptr;
    }

    bool success = true;

    auto extract_tensor = [&](int data_idx, int dims_idx, sd_tensor_raw_t& raw) {
        if (!success) return;
        jfloatArray dataArr = static_cast<jfloatArray>(env->GetObjectArrayElement(condArr, data_idx));
        jintArray dimsArr = static_cast<jintArray>(env->GetObjectArrayElement(condArr, dims_idx));

        if (dataArr && dimsArr) {
            const jsize dataLen = env->GetArrayLength(dataArr);
            const jsize dimsLen = env->GetArrayLength(dimsArr);

            raw.ndims = std::min(static_cast<int>(dimsLen), 4);
            jint* dims = env->GetIntArrayElements(dimsArr, nullptr);
            int64_t expected_len = 1;
            for (int i = 0; i < raw.ndims; ++i) {
                raw.ne[i] = dims[i];
                expected_len *= dims[i];
            }
            ALOGI("Reconstructed raw tensor: ndims=%d, ne=[%lld, %lld, %lld, %lld]",
                  raw.ndims, (long long)raw.ne[0], (long long)raw.ne[1], (long long)raw.ne[2], (long long)raw.ne[3]);
            env->ReleaseIntArrayElements(dimsArr, dims, JNI_ABORT);

            if (expected_len != dataLen) {
                throwJavaException(env, "java/lang/IllegalArgumentException", "Dimension product mismatch with data length");
                success = false;
                if (dataArr) env->DeleteLocalRef(dataArr);
                if (dimsArr) env->DeleteLocalRef(dimsArr);
                return;
            }

            raw.data = static_cast<float*>(malloc(dataLen * sizeof(float)));
            if (!raw.data) {
                throwJavaException(env, "java/lang/OutOfMemoryError", "Failed to allocate memory for tensor data");
                success = false;
                if (dataArr) env->DeleteLocalRef(dataArr);
                if (dimsArr) env->DeleteLocalRef(dimsArr);
                return;
            }

            jfloat* data = env->GetFloatArrayElements(dataArr, nullptr);
            memcpy(raw.data, data, dataLen * sizeof(float));
            env->ReleaseFloatArrayElements(dataArr, data, JNI_ABORT);
        } else {
            raw.ndims = 0;
            raw.data = nullptr;
        }

        if (dataArr) env->DeleteLocalRef(dataArr);
        if (dimsArr) env->DeleteLocalRef(dimsArr);
    };

    extract_tensor(0, 1, cond->c_crossattn);
    if (!success) {
        sd_free_condition(cond);
        return nullptr;
    }
    extract_tensor(2, 3, cond->c_vector);
    if (!success) {
        sd_free_condition(cond);
        return nullptr;
    }
    extract_tensor(4, 5, cond->c_concat);
    if (!success) {
        sd_free_condition(cond);
        return nullptr;
    }

    return cond;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativePrecomputeCondition(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height, jint clipSkip,
        jboolean jIsVideo) {
    (void)thiz;

    // Standard UTF-8 (not GetStringUTFChars' Modified UTF-8) so emoji and other
    // supplementary characters reach the native tokenizer as valid bytes.
    const std::string promptUtf8 = llmedge_jstring_to_utf8(env, jPrompt);
    const std::string negativeUtf8 = llmedge_jstring_to_utf8(env, jNegative);
    const char* prompt = promptUtf8.c_str();
    const char* negative = negativeUtf8.c_str();

    auto releaseStrings = [&]() {
    };

    sd_condition_raw_t* cond = nullptr;

    if (handlePtr != 0) {
        auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
        SdResolvedPromptLoras resolved = resolve_prompt_loras(prompt, negative, handle->loraModelDir);
        if (handle->ctx) {
            try {
                cond = sd_precompute_condition(handle->ctx, resolved.prompt.c_str(), clipSkip, width, height, jIsVideo == JNI_TRUE);
            } catch (const std::exception& e) {
                releaseStrings();
                throwJavaException(env, "java/lang/RuntimeException", e.what());
                return nullptr;
            }
        }
#ifndef SD_JNI_TESTING
        else if (handle->t5_ctx) {
            auto* t5 = static_cast<T5CLIPEmbedder*>(handle->t5_ctx);
            try {
                ConditionerParams cparams;
                cparams.text = resolved.prompt.c_str();
                cparams.clip_skip = clipSkip;
                cparams.width = width;
                cparams.height = height;
                cparams.zero_out_masked = (jIsVideo == JNI_TRUE);

                SDCondition sd_cond =
                        t5->get_learned_condition(sd_get_num_physical_cores_safe(), cparams);

                cond = static_cast<sd_condition_raw_t*>(calloc(1, sizeof(sd_condition_raw_t)));
                if (!cond) {
                    throw std::runtime_error("Out of memory allocating condition");
                }

                auto tensor_to_raw_f32 = [](const sd::Tensor<float>& t, sd_tensor_raw_t& raw) {
                    if (t.empty()) return;
                    raw.ndims = (int)t.dim();
                    for (int i = 0; i < t.dim(); ++i) raw.ne[i] = t.shape()[i];
                    ALOGI("Precomputed condition tensor to raw (T5): ndims=%d, shape=[%lld, %lld, %lld, %lld]",
                          raw.ndims, (long long)raw.ne[0], (long long)raw.ne[1], (long long)raw.ne[2], (long long)raw.ne[3]);
                    const size_t n = static_cast<size_t>(t.numel());
                    raw.data = static_cast<float*>(malloc(sizeof(float) * n));
                    if (!raw.data) {
                        raw.ndims = 0;
                        return;
                    }
                    memcpy(raw.data, t.data(), sizeof(float) * n);
                };

                if (!sd_cond.c_crossattn.empty()) tensor_to_raw_f32(sd_cond.c_crossattn, cond->c_crossattn);
                if (!sd_cond.c_vector.empty()) tensor_to_raw_f32(sd_cond.c_vector, cond->c_vector);
                if (!sd_cond.c_concat.empty()) tensor_to_raw_f32(sd_cond.c_concat, cond->c_concat);

            } catch (const std::exception& e) {
                releaseStrings();
                throwJavaException(env, "java/lang/RuntimeException", e.what());
                return nullptr;
            }
        } else if (handle->llm_ctx) {
            // FLUX.2 sequential mode: encoder-only (Qwen3) handle precomputes the conditioning.
            auto* llm = static_cast<LLMEmbedder*>(handle->llm_ctx);
            try {
                ConditionerParams cparams;
                cparams.text      = resolved.prompt.c_str();
                cparams.clip_skip = clipSkip;
                cparams.width     = width;
                cparams.height    = height;
                cparams.zero_out_masked = (jIsVideo == JNI_TRUE);

                SDCondition sd_cond =
                        llm->get_learned_condition(sd_get_num_physical_cores_safe(), cparams);

                cond = static_cast<sd_condition_raw_t*>(calloc(1, sizeof(sd_condition_raw_t)));
                if (!cond) {
                    throw std::runtime_error("Out of memory allocating condition");
                }

                auto tensor_to_raw_f32 = [](const sd::Tensor<float>& t, sd_tensor_raw_t& raw) {
                    if (t.empty()) return;
                    raw.ndims = (int)t.dim();
                    for (int i = 0; i < t.dim(); ++i) raw.ne[i] = t.shape()[i];
                    ALOGI("Precomputed condition tensor to raw (LLM): ndims=%d, shape=[%lld, %lld, %lld, %lld]",
                          raw.ndims, (long long)raw.ne[0], (long long)raw.ne[1], (long long)raw.ne[2], (long long)raw.ne[3]);
                    const size_t n = static_cast<size_t>(t.numel());
                    raw.data = static_cast<float*>(malloc(sizeof(float) * n));
                    if (!raw.data) {
                        raw.ndims = 0;
                        return;
                    }
                    memcpy(raw.data, t.data(), sizeof(float) * n);
                };

                if (!sd_cond.c_crossattn.empty()) tensor_to_raw_f32(sd_cond.c_crossattn, cond->c_crossattn);
                if (!sd_cond.c_vector.empty()) tensor_to_raw_f32(sd_cond.c_vector, cond->c_vector);
                if (!sd_cond.c_concat.empty()) tensor_to_raw_f32(sd_cond.c_concat, cond->c_concat);

            } catch (const std::exception& e) {
                releaseStrings();
                throwJavaException(env, "java/lang/RuntimeException", e.what());
                return nullptr;
            }
        }
#endif
        else {
            throwJavaException(env, "java/lang/IllegalStateException", "Invalid handle state");
            releaseStrings();
            return nullptr;
        }
    } else {
        throwJavaException(env, "java/lang/IllegalStateException", "StableDiffusion not initialized");
        releaseStrings();
        return nullptr;
    }

    releaseStrings();

    if (!cond) {
        throwJavaException(env, "java/lang/IllegalStateException", "Condition precompute failed");
        return nullptr;
    }

    jclass objClass = env->FindClass("java/lang/Object");
    if (!objClass) {
        sd_free_condition(cond);
        throwJavaException(env, "java/lang/RuntimeException", "Unable to find java/lang/Object");
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(6, objClass, nullptr);
    if (!result) {
        sd_free_condition(cond);
        throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate result array");
        return nullptr;
    }

    auto push_tensor = [&](const sd_tensor_raw_t* t, int data_index, int dims_index) {
        if (t == nullptr || t->ndims == 0 || t->data == nullptr) {
            env->SetObjectArrayElement(result, data_index, nullptr);
            env->SetObjectArrayElement(result, dims_index, nullptr);
            return;
        }
        size_t count = 1;
        for (int i = 0; i < t->ndims; ++i) {
            count *= static_cast<size_t>(t->ne[i]);
        }

        jfloatArray floatArr = env->NewFloatArray(static_cast<jsize>(count));
        if (!floatArr) {
            env->SetObjectArrayElement(result, data_index, nullptr);
        } else {
            env->SetFloatArrayRegion(floatArr, 0, static_cast<jsize>(count), reinterpret_cast<const jfloat*>(t->data));
            env->SetObjectArrayElement(result, data_index, floatArr);
            env->DeleteLocalRef(floatArr);
        }

        jintArray dimsArr = env->NewIntArray(t->ndims);
        if (!dimsArr) {
            env->SetObjectArrayElement(result, dims_index, nullptr);
        } else {
            jint dims[4] = {0, 0, 0, 0};
            for (int i = 0; i < t->ndims && i < 4; ++i) {
                dims[i] = t->ne[i];
            }
            env->SetIntArrayRegion(dimsArr, 0, t->ndims, dims);
            env->SetObjectArrayElement(result, dims_index, dimsArr);
            env->DeleteLocalRef(dimsArr);
        }
    };

    push_tensor(&cond->c_crossattn, 0, 1);
    push_tensor(&cond->c_vector, 2, 3);
    push_tensor(&cond->c_concat, 4, 5);

    sd_free_condition(cond);
    return result;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2ImgWithPrecomputedCondition(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint steps, jfloat cfg, jlong seed,
        jboolean jVaeTiling,
        jobjectArray condArr, jobjectArray uncondArr,
        jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent) {
    (void)thiz;
    if (handlePtr == 0) {
        ALOGE("StableDiffusion not initialized");
        return nullptr;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);

    if (!handle->ctx) {
        throwJavaException(env, "java/lang/IllegalStateException", "StableDiffusion context is null");
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

    auto releaseStrings = [&]() {
    };

    sd_condition_raw_t* cond = reconstruct_condition(env, condArr);
    sd_condition_raw_t* uncond = reconstruct_condition(env, uncondArr);

    if (!cond) {
        ALOGE("Failed to reconstruct condition");
        if (uncond) sd_free_condition(uncond);
        releaseStrings();
        return nullptr;
    }

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
    try {
        out = sd_generate_image_with_precomputed_condition(handle->ctx, &gen, cond, uncond);
    } catch (const std::exception& e) {
        releaseStrings();
        sd_free_condition(cond);
        if (uncond) sd_free_condition(uncond);
        const char* clazz = handle->cancellationRequested.load()
                ? "java/util/concurrent/CancellationException"
                : "java/lang/RuntimeException";
        throwJavaException(env, clazz, e.what());
        return nullptr;
    }

    releaseStrings();
    sd_free_condition(cond);
    if (uncond) sd_free_condition(uncond);

    if (!out || !out[0].data) {
        ALOGE("generate_image failed");
        if (out) free(out);
        handle->cancellationRequested.store(false);
        return nullptr;
    }

    const size_t byteCount = static_cast<size_t>(out[0].width) * out[0].height * out[0].channel;
    jbyteArray jbytes = env->NewByteArray(static_cast<jsize>(byteCount));
    if (!jbytes) {
        free(out[0].data);
        free(out);
        handle->cancellationRequested.store(false);
        return nullptr;
    }
    env->SetByteArrayRegion(jbytes, 0, static_cast<jsize>(byteCount), reinterpret_cast<jbyte*>(out[0].data));

    free(out[0].data);
    free(out);

    handle->cancellationRequested.store(false);
    return jbytes;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2VidWithPrecomputedCondition(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint videoFrames, jint steps, jfloat cfg, jlong seed,
        jint jSampleMethod, jint jScheduler, jfloat jStrength,
        jbyteArray jInitImage, jint initWidth, jint initHeight,
        jobjectArray condArr, jobjectArray uncondArr,
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
        throwJavaException(env, "java/lang/IllegalStateException", "StableDiffusion context is null");
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

    sd_condition_raw_t* cond_use = reconstruct_condition(env, condArr);
    sd_condition_raw_t* uncond_use = reconstruct_condition(env, uncondArr);

    handle->stepsPerFrame = 0;
    handle->totalSteps = sample.sample_steps > 0 ? sample.sample_steps : 0;

    SdProgressCallbackGuard callbackGuard(handle);

    sd_image_t* frames = nullptr;
    int numFrames = 0;
    try {
        ALOGI("Calling sd_generate_video_with_precomputed_condition...");
        frames = sd_generate_video_with_precomputed_condition(handle->ctx, &gen, cond_use, uncond_use, &numFrames);
        ALOGI("sd_generate_video_with_precomputed_condition returned %d frames", numFrames);
    } catch (const std::exception& e) {
        ALOGE("Exception in sd_generate_video_with_precomputed_condition: %s", e.what());
        releaseStrings();
        if (cond_use) sd_free_condition(cond_use);
        if (uncond_use) sd_free_condition(uncond_use);
        const char* clazz = handle->cancellationRequested.load()
                ? "java/util/concurrent/CancellationException"
                : "java/lang/RuntimeException";
        throwJavaException(env, clazz, e.what());
        return nullptr;
    }

    releaseStrings();

    if (cond_use) sd_free_condition(cond_use);
    if (uncond_use) sd_free_condition(uncond_use);

    if (!frames || numFrames <= 0) {
        if (frames) free_sd_generated_frames(frames, numFrames);
        throwJavaException(env, "java/lang/IllegalStateException", "Video generation failed");
        return nullptr;
    }

    jobjectArray result = convert_sd_frames_to_java(env, frames, numFrames);
    handle->cancellationRequested.store(false);
    return result;
}
