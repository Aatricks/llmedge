#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <algorithm>

#include "jni_thread_cache.h"

#if __has_include(<android/log.h>)
#include <android/log.h>
#else
#include <cstdio>
#include <cstdarg>
#define ANDROID_LOG_DEBUG 3
#define ANDROID_LOG_INFO 4
#define ANDROID_LOG_WARN 5
#define ANDROID_LOG_ERROR 6
inline int __android_log_print(int level, const char* tag, const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[%s] ", tag);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(args);

    // Also print to stdout for good measure
    va_start(args, format);
    fprintf(stdout, "[%s] ", tag);
    vfprintf(stdout, format, args);
    fprintf(stdout, "\n");
    fflush(stdout);
    va_end(args);
    return 0;
}
#endif

#define GGML_MAX_NAME 128
#include "stable-diffusion.h"
#include "sd_jni_internal.h"
#if defined(SD_USE_VULKAN)
#include "ggml-vulkan.h"
#endif
#include "model.h"
#include "conditioner.hpp"
#include "ggml-backend.h"

#define LOG_TAG "SmolSD"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

// --------------------------------------------------------------------------------------
// Upstream stable-diffusion.cpp enum compatibility
//
// llmedge's Kotlin enums intentionally include DEFAULT=0, then the upstream enum values.
// Upstream stable-diffusion.cpp does NOT include DEFAULT; instead, callers should pass
// *_COUNT to request the model-specific default.
// --------------------------------------------------------------------------------------

static inline int sd_get_num_physical_cores_safe() {
    const int32_t n = sd_get_num_physical_cores();
    return n > 0 ? (int)n : 1;
}

static inline bool map_sample_method_from_kotlin_id(int kotlin_id, enum sample_method_t* out) {
    if (!out) return false;
    // Kotlin ids:
    //   0=DEFAULT, 1=EULER, 2=HEUN, 3=DPM2, 4=DPMPP2S_A, 5=DPMPP2M, 6=DPMPP2MV2,
    //   7=IPNDM, 8=IPNDM_V, 9=LCM, 10=DDIM_TRAILING, 11=TCD, 12=EULER_A
    // Upstream enum sample_method_t:
    //   0=EULER, 1=EULER_A, 2=HEUN, 3=DPM2, 4=DPMPP2S_A, 5=DPMPP2M, 6=DPMPP2Mv2,
    //   7=IPNDM, 8=IPNDM_V, 9=LCM, 10=DDIM_TRAILING, 11=TCD
    switch (kotlin_id) {
        case 1:  *out = EULER_SAMPLE_METHOD; break;
        case 12: *out = EULER_A_SAMPLE_METHOD; break;
        case 2:  *out = HEUN_SAMPLE_METHOD; break;
        case 3:  *out = DPM2_SAMPLE_METHOD; break;
        case 4:  *out = DPMPP2S_A_SAMPLE_METHOD; break;
        case 5:  *out = DPMPP2M_SAMPLE_METHOD; break;
        case 6:  *out = DPMPP2Mv2_SAMPLE_METHOD; break;
        case 7:  *out = IPNDM_SAMPLE_METHOD; break;
        case 8:  *out = IPNDM_V_SAMPLE_METHOD; break;
        case 9:  *out = LCM_SAMPLE_METHOD; break;
        case 10: *out = DDIM_TRAILING_SAMPLE_METHOD; break;
        case 11: *out = TCD_SAMPLE_METHOD; break;
        default:
            return false;
    }
    return true;
}

static inline bool map_scheduler_from_kotlin_id(int kotlin_id, enum scheduler_t* out) {
    if (!out) return false;
    // Kotlin ids: 0=DEFAULT, 1=DISCRETE, 2=KARRAS, ...
    // Upstream scheduler_t has no DEFAULT; use SCHEDULER_COUNT to request default.
    if (kotlin_id <= 0) return false;
    const int upstream = kotlin_id - 1;
    if (upstream < 0 || upstream >= (int)SCHEDULER_COUNT) return false;
    *out = (enum scheduler_t)upstream;
    return true;
}

SD_JNI_INTERNAL void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    if (!env) return;
    jclass exClass = env->FindClass(className);
    if (!exClass) return;
    env->ThrowNew(exClass, message);
}

SD_JNI_INTERNAL void clearProgressCallback(JNIEnv* env, SdHandle* handle) {
    if (!handle) return;
    if (handle->progressCallbackGlobalRef && env) {
        env->DeleteGlobalRef(handle->progressCallbackGlobalRef);
    }
    handle->progressCallbackGlobalRef = nullptr;
    handle->progressMethodID = nullptr;
    handle->currentFrame = 0;
    handle->totalFrames = 0;
    handle->stepsPerFrame = 0;
    handle->totalSteps = 0;
    handle->cancellationRequested.store(false);
}

SD_JNI_INTERNAL void sd_video_progress_wrapper(int step, int steps, float time, void* data) {
    auto* handle = static_cast<SdHandle*>(data);
    if (!handle) return;

    if (handle->cancellationRequested.load()) {
        throw std::runtime_error("Video generation cancelled");
    }

    if (!handle->progressCallbackGlobalRef || !handle->jvm || !handle->progressMethodID) {
        return;
    }

    JNIEnv* env = jni_thread_cache_get_env();
    if (!env) return;

    const int totalFrames = handle->totalFrames > 0 ? handle->totalFrames : 1;
    // For video sampling, stable-diffusion.cpp reports progress per sampling step over the
    // whole clip (not "steps per frame"). Prefer the callback-provided total when available.
    const int totalSteps = steps > 0 ? steps : (handle->totalSteps > 0 ? handle->totalSteps : 1);

    // Provide a best-effort "frame" value for UI/clients by mapping step progress onto
    // the frame index range. This is not a true per-frame denoise progress.
    int inferredFrame = 0;
    if (totalFrames > 1 && totalSteps > 0) {
        inferredFrame = (step * totalFrames) / totalSteps;
        if (inferredFrame < 0) inferredFrame = 0;
        if (inferredFrame >= totalFrames) inferredFrame = totalFrames - 1;
    }
    handle->currentFrame = inferredFrame;

    env->CallVoidMethod(
            handle->progressCallbackGlobalRef,
            handle->progressMethodID,
            static_cast<jint>(std::min(step, totalSteps)),
            static_cast<jint>(totalSteps),
            static_cast<jint>(handle->currentFrame),
            static_cast<jint>(handle->totalFrames),
            static_cast<jfloat>(time));

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeCheckBindings(JNIEnv*, jclass) {
    return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeGetVulkanDeviceCount(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
#ifdef SD_USE_VULKAN
    return (jint)ggml_backend_vk_get_device_count();
#else
    return 0;
#endif
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeGetVulkanDeviceMemory(JNIEnv* env, jclass clazz, jint deviceIndex) {
    (void)clazz;
#ifdef SD_USE_VULKAN
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_vk_get_device_memory((int)deviceIndex, &free_mem, &total_mem);
    jlongArray arr = env->NewLongArray(2);
    if (!arr) return nullptr;
    jlong vals[2];
    vals[0] = (jlong)free_mem;
    vals[1] = (jlong)total_mem;
    env->SetLongArrayRegion(arr, 0, 2, vals);
    return arr;
#else
    jlongArray arr = env->NewLongArray(2);
    if (!arr) return nullptr;
    jlong vals[2] = {0, 0};
    env->SetLongArrayRegion(arr, 0, 2, vals);
    return arr;
#endif
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeEstimateModelParamsMemory(JNIEnv* env, jclass clazz, jstring jModelPath, jint deviceIndex) {
    (void)clazz;
    if (!jModelPath) return (jlong)-1;
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    if (!modelPath) return (jlong)-1;
    ModelLoader model_loader;
    bool ok = model_loader.init_from_file(modelPath);
    if (!ok) {
        env->ReleaseStringUTFChars(jModelPath, modelPath);
        return (jlong)-1;
    }
    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (deviceIndex >= 0 && ggml_backend_vk_get_device_count() > deviceIndex) {
        backend = ggml_backend_vk_init(deviceIndex);
    }
#endif
    int64_t params_mem = model_loader.get_params_mem_size(backend, GGML_TYPE_COUNT);
    if (backend) ggml_backend_free(backend);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    return (jlong)params_mem;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeEstimateModelParamsMemoryDetailed(JNIEnv* env, jclass clazz, jstring jModelPath, jint deviceIndex) {
    (void)clazz;
    if (!jModelPath) return nullptr;
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    if (!modelPath) return nullptr;
    ModelLoader model_loader;
    bool ok = model_loader.init_from_file(modelPath);
    if (!ok) {
        env->ReleaseStringUTFChars(jModelPath, modelPath);
        return nullptr;
    }
    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (deviceIndex >= 0 && ggml_backend_vk_get_device_count() > deviceIndex) {
        backend = ggml_backend_vk_init(deviceIndex);
    }
#endif
    // Detailed per-prefix parameter memory estimation API was removed upstream.
    // As a best-effort fallback, return zeros for subcomponents and provide the total.
    // Callers currently only rely on the total to make offload decisions.
    jlong clip = (jlong)0;
    jlong diffusion = (jlong)0;
    jlong vae = (jlong)0;
    jlong control = (jlong)0;
    jlong pmid = (jlong)0;
    jlong total = (jlong)model_loader.get_params_mem_size(backend, GGML_TYPE_COUNT);
    if (backend) ggml_backend_free(backend);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    jlong vals[6] = {clip, diffusion, vae, control, pmid, total};
    jlongArray arr = env->NewLongArray(6);
    if (!arr) return nullptr;
    env->SetLongArrayRegion(arr, 0, 6, vals);
    return arr;
}

static void sd_android_log_cb(enum sd_log_level_t level, const char* text, void* data) {
    (void)data;
    if (!text) return;
    switch (level) {
        case SD_LOG_DEBUG: __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s", text); break;
        case SD_LOG_INFO:  __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, "%s", text); break;
        case SD_LOG_WARN:  __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, "%s", text); break;
        case SD_LOG_ERROR: __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "%s", text); break;
        default: __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "%s", text); break;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeCreate(
    JNIEnv* env, jclass clazz,
        jstring jModelPath,
        jstring jVaePath,
        jstring jT5xxlPath,
        jstring jTaesdPath,
        jint nThreads,
        jboolean offloadToCpu,
        jboolean keepClipOnCpu,
        jboolean keepVaeOnCpu,
        jboolean flashAttn,
        jboolean jvaeDecodeOnly,
        jfloat flowShift,
        jstring jLoraModelDir, jint jLoraApplyMode) {
    (void)clazz;
    FILE* f = fopen("/tmp/sdcpp_log.txt", "a");
    if (f) {
        fprintf(f, "[SmolSD] nativeCreate ENTERED\n");
        fclose(f);
    }
    fprintf(stderr, "[SmolSD] nativeCreate ENTERED\n"); fflush(stderr);
    const char* modelPath = jModelPath ? env->GetStringUTFChars(jModelPath, nullptr) : nullptr;
    const char* vaePath   = jVaePath   ? env->GetStringUTFChars(jVaePath,   nullptr) : nullptr;
    const char* t5xxlPath = jT5xxlPath ? env->GetStringUTFChars(jT5xxlPath, nullptr) : nullptr;
    const char* taesdPath = jTaesdPath ? env->GetStringUTFChars(jTaesdPath, nullptr) : nullptr;

    sd_set_log_callback(sd_android_log_cb, nullptr);

    ALOGI("Initializing Stable Diffusion with:");
    ALOGI("  modelPath=%s", modelPath ? modelPath : "NULL");
    ALOGI("  vaePath=%s", vaePath ? vaePath : "NULL");
    ALOGI("  t5xxlPath=%s", t5xxlPath ? t5xxlPath : "NULL");
    ALOGI("  taesdPath=%s", taesdPath ? taesdPath : "NULL");
    ALOGI("  offloadToCpu=%s, keepClipOnCpu=%s, keepVaeOnCpu=%s, flashAttn=%s, vaeDecodeOnly=%s",
          offloadToCpu ? "true" : "false",
          keepClipOnCpu ? "true" : "false",
          keepVaeOnCpu ? "true" : "false",
          flashAttn ? "true" : "false",
          jvaeDecodeOnly ? "true" : "false");

    sd_ctx_params_t p{};
    sd_ctx_params_init(&p);
    p.model_path = modelPath ? modelPath : "";
    p.vae_path = vaePath ? vaePath : "";
    // Pass through nullptr when no T5XXL path is provided. Some pipelines
    // distinguish between null and empty string; using nullptr ensures the
    // text encoder is selected correctly for SD 1.x models.
    p.t5xxl_path = t5xxlPath; // keep null if not provided
    p.taesd_path = taesdPath ? taesdPath : "";
    p.free_params_immediately = true;
    p.n_threads = nThreads > 0 ? nThreads : sd_get_num_physical_cores_safe();
    p.offload_params_to_cpu = offloadToCpu;
    p.keep_clip_on_cpu = keepClipOnCpu;
    p.keep_vae_on_cpu = keepVaeOnCpu;
    p.diffusion_flash_attn = flashAttn;
    p.flow_shift = flowShift;
    // VAE decode only (usually true for TAE, false for full VAE if encoding needed for I2V)
    p.vae_decode_only = jvaeDecodeOnly;
    if (jLoraModelDir) {
        const char* loraPath = env->GetStringUTFChars(jLoraModelDir, nullptr);
        if (loraPath) {
            p.lora_model_dir = strdup(loraPath);
            env->ReleaseStringUTFChars(jLoraModelDir, loraPath);
        }
    }
    p.lora_apply_mode = static_cast<enum lora_apply_mode_t>(jLoraApplyMode);

    sd_ctx_t* ctx = new_sd_ctx(&p);

    if (!ctx) {
        // Fallback: Check if we can load as T5-only context
        // This is a hack to support sequential loading where we only want the text encoder
        if (modelPath && !vaePath && !t5xxlPath) {
             ALOGI("Attempting to load as T5-only context: %s", modelPath);
             // We need to manually instantiate T5CLIPEmbedder
             // But we need a backend.
             // And ModelLoader.

             // We can't easily do this inside nativeCreate because we need to return a jlong handle
             // and we need to store the T5 object in it.

             // Let's try to instantiate T5CLIPEmbedder here.
             ModelLoader model_loader;
             if (model_loader.init_from_file(modelPath, "text_encoders.t5xxl.transformer.")) {
                 ALOGI("ModelLoader initialized for T5");
                 model_loader.convert_tensors_name();

                 ggml_backend_t backend = nullptr;
                 #ifdef SD_USE_VULKAN
                 if (ggml_backend_vk_get_device_count() > 0) {
                     backend = ggml_backend_vk_init(0);
                 }
                 #endif
                 if (!backend) {
                     backend = ggml_backend_cpu_init();
                 }

                 if (!backend) {
                     ALOGE("Vulkan backend not available and CPU backend init failed/missing");
                     if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                     if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                     if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                     if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
                     return 0;
                 }
                 ALOGI("Backend initialized for T5");

                 // T5CLIPEmbedder(backend, offload, storage, use_mask, mask_pad, is_umt5)
                 bool is_umt5 = std::string(modelPath).find("umt5") != std::string::npos;
                 ALOGI("Creating T5CLIPEmbedder (is_umt5=%d)", is_umt5);

                 auto* t5 = new T5CLIPEmbedder(backend, offloadToCpu, model_loader.get_tensor_storage_map(), false, 0, is_umt5);
                 ALOGI("Allocating params buffer for T5");
                 t5->alloc_params_buffer();

                 // Load weights
                 std::map<std::string, struct ggml_tensor*> tensors;
                 t5->get_param_tensors(tensors);
                 ALOGI("Got param tensors for T5: %zu tensors", tensors.size());

                 std::set<std::string> ignore_tensors;
                 ALOGI("Loading tensors for T5");
                 model_loader.load_tensors(tensors, ignore_tensors, sd_get_num_physical_cores_safe());

                 auto* handle = new SdHandle();
                 handle->ctx = nullptr;
                 handle->t5_ctx = t5;
                 if (env) {
                     env->GetJavaVM(&handle->jvm);
                     jni_thread_cache_init(handle->jvm);
                 }
                 ALOGI("T5-only context created successfully");

                 if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                 if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                 if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                 if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);

                 return reinterpret_cast<jlong>(handle);
             } else {
                 ALOGE("Failed to init ModelLoader for T5");
             }
        }

        ALOGE("Failed to create sd_ctx");
        if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
        if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
        if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
        if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
        return 0;
    }

    if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
    if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
    if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
    if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);

    auto* handle = new SdHandle();
    handle->ctx = ctx;
    if (env) {
        env->GetJavaVM(&handle->jvm);
        jni_thread_cache_init(handle->jvm);
    }
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(JNIEnv* env, jobject, jlong handlePtr) {
    if (handlePtr == 0) return;
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    clearProgressCallback(env, handle);
    sd_set_progress_callback(nullptr, nullptr);
    if (handle->ctx) {
        free_sd_ctx(handle->ctx);
        handle->ctx = nullptr;
    }
    if (handle->t5_ctx) {
        auto* t5 = static_cast<T5CLIPEmbedder*>(handle->t5_ctx);
        t5->free_params_buffer();
        delete t5;
        handle->t5_ctx = nullptr;
    }
    delete handle;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeIsEasyCacheSupported(JNIEnv* env, jobject, jlong handlePtr) {
    (void)env;
    (void)handlePtr;
    // Upstream stable-diffusion.cpp does not expose a public C API to query EasyCache
    // support. The generation APIs will internally enable/disable based on model type.
    // Return false conservatively to prevent callers from assuming support.
    return JNI_FALSE;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Img(
    JNIEnv* env, jobject thiz, jlong handlePtr,
    jstring jPrompt, jstring jNegative,
    jint width, jint height,
    jint steps, jfloat cfg, jlong seed,
    jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent) {
    (void)thiz;
    if (handlePtr == 0) {
        ALOGE("StableDiffusion not initialized");
        return nullptr;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    if (!handle->ctx) {
        // This can happen when the handle was created in "T5-only" mode for sequential
        // precompute, but the caller mistakenly invoked txt2img.
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

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.prompt = prompt;
    gen.negative_prompt = negative;
    gen.width = width;
    gen.height = height;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.batch_count = 1;
    gen.easycache.enabled = jEasyCacheEnabled ? true : false;
    gen.easycache.reuse_threshold = (float)jEasyCacheReuseThreshold;
    gen.easycache.start_percent = (float)jEasyCacheStartPercent;
    gen.easycache.end_percent = (float)jEasyCacheEndPercent;

    sd_image_t* out = generate_image(handle->ctx, &gen);

    if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
    if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);

    if (!out || !out[0].data) {
        ALOGE("generate_image failed");
        return nullptr;
    }

    const size_t byteCount = (size_t)out[0].width * out[0].height * out[0].channel;
    jbyteArray jbytes = env->NewByteArray((jsize)byteCount);
    if (!jbytes) {
        free(out[0].data);
        free(out);
        return nullptr;
    }
    env->SetByteArrayRegion(jbytes, 0, (jsize)byteCount, reinterpret_cast<jbyte*>(out[0].data));

    // Free native buffers allocated by stable-diffusion.cpp
    free(out[0].data);
    free(out);

    return jbytes;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Vid(
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

    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    auto releaseStrings = [&]() {
        if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
        if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
    };

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    if (cfg > 0.f) sample.guidance.txt_cfg = cfg;

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

    // Map Kotlin enums (DEFAULT=0) to upstream enums (no DEFAULT).
    // Use *_COUNT as a sentinel to request model defaults.
    {
        enum sample_method_t mapped_method;
        if (map_sample_method_from_kotlin_id((int)jSampleMethod, &mapped_method)) {
            gen.sample_params.sample_method = mapped_method;
        } else {
            gen.sample_params.sample_method = SAMPLE_METHOD_COUNT;
        }

        enum scheduler_t mapped_sched;
        if (map_scheduler_from_kotlin_id((int)jScheduler, &mapped_sched)) {
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
            env->GetByteArrayRegion(jInitImage, 0, initSize,
                                    reinterpret_cast<jbyte*>(initImageData.data()));
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

    // Set easycache parameters for both T2V and I2V modes
    gen.easycache.enabled = jEasyCacheEnabled ? true : false;
    gen.easycache.reuse_threshold = (float)jEasyCacheReuseThreshold;
    gen.easycache.start_percent = (float)jEasyCacheStartPercent;
    gen.easycache.end_percent = (float)jEasyCacheEndPercent;

    // Video sampling steps are global for the clip; do not multiply by frame count.
    handle->stepsPerFrame = 0;
    handle->totalSteps = sample.sample_steps > 0 ? sample.sample_steps : 0;

    // Ensure progress callback is wired for cancellation even if Kotlin-side callback is null.
    if (!handle->progressCallbackGlobalRef) {
        sd_set_progress_callback(sd_video_progress_wrapper, handle);
    }

    sd_image_t* frames = nullptr;
    int numFrames = 0;
    try {
        frames = generate_video(handle->ctx, &gen, &numFrames);
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
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
        }
        throwJavaException(env, "java/lang/IllegalStateException", "Video generation failed");
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    jclass byteArrayClass = env->FindClass("[B");
    if (!byteArrayClass) {
        for (int i = 0; i < numFrames; ++i) {
            if (frames[i].data) {
                sd_jni_notify_frame_buffer_freed(frames[i].data);
                free(frames[i].data);
            }
        }
        sd_jni_notify_frame_array_freed(frames);
        free(frames);
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(numFrames, byteArrayClass, nullptr);
    if (!result) {
        for (int i = 0; i < numFrames; ++i) {
            if (frames[i].data) {
                sd_jni_notify_frame_buffer_freed(frames[i].data);
                free(frames[i].data);
            }
        }
        sd_jni_notify_frame_array_freed(frames);
        free(frames);
        throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate video frame array");
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    for (int i = 0; i < numFrames; ++i) {
        if (!frames[i].data) {
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            throwJavaException(env, "java/lang/IllegalStateException", "Missing frame data");
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        const size_t byteCount = static_cast<size_t>(frames[i].width) * frames[i].height * frames[i].channel;
        jbyteArray frameBytes = env->NewByteArray(static_cast<jsize>(byteCount));
        if (!frameBytes) {
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate frame buffer");
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        env->SetByteArrayRegion(frameBytes, 0, static_cast<jsize>(byteCount),
                                 reinterpret_cast<jbyte*>(frames[i].data));
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(frameBytes);
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        env->SetObjectArrayElement(result, i, frameBytes);
        env->DeleteLocalRef(frameBytes);
        sd_jni_notify_frame_buffer_freed(frames[i].data);
        free(frames[i].data);
    }

    sd_jni_notify_frame_array_freed(frames);
    free(frames);
    if (!handle->progressCallbackGlobalRef) {
        sd_set_progress_callback(nullptr, nullptr);
    }
    handle->cancellationRequested.store(false);
    return result;
}

// -----------------------------------------------------------------------------
// Precomputed-condition APIs
//
// llmedge uses these bindings to support sequential loading/unloading on
// memory-constrained devices:
//   1) load T5 -> precompute condition -> unload T5
//   2) load diffusion+VAE -> generate using precomputed condition
// -----------------------------------------------------------------------------

// JNI wrapper: precompute condition for a given prompt & video params
extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativePrecomputeCondition(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height, jint clipSkip) {
    (void)thiz;

    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    auto releaseStrings = [&]() {
        if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
        if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
    };

    sd_condition_raw_t* cond = nullptr;

    if (handlePtr != 0) {
        // Legacy path: use existing context
        auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
        if (handle->ctx) {
            try {
                cond = sd_precompute_condition(handle->ctx,
                                               prompt,
                                               clipSkip,
                                               width,
                                               height,
                                               true);
            } catch (const std::exception& e) {
                releaseStrings();
                throwJavaException(env, "java/lang/RuntimeException", e.what());
                return nullptr;
            }
        } else if (handle->t5_ctx) {
            // T5-only path
            auto* t5 = static_cast<T5CLIPEmbedder*>(handle->t5_ctx);
            try {
                // T5CLIPEmbedder::get_learned_condition returns SDCondition
                // SDCondition has ggml_tensor* c_crossattn, c_vector, c_concat

                // We need a work context
                struct ggml_init_params params;
                params.mem_size   = 1024 * 1024 * 1024; // 1GB for intermediate tensors?
                params.mem_buffer = nullptr;
                params.no_alloc   = false;
                struct ggml_context* work_ctx = ggml_init(params);

                ConditionerParams cparams;
                cparams.text = prompt;
                cparams.clip_skip = clipSkip;
                cparams.width = width;
                cparams.height = height;

                SDCondition sd_cond = t5->get_learned_condition(work_ctx, sd_get_num_physical_cores_safe(), cparams);

                // Convert SDCondition to sd_condition_raw_t (always float32)
                cond = (sd_condition_raw_t*)calloc(1, sizeof(sd_condition_raw_t));
                if (!cond) {
                    ggml_free(work_ctx);
                    throw std::runtime_error("Out of memory allocating condition");
                }

                auto tensor_to_raw_f32 = [](struct ggml_tensor* t, sd_tensor_raw_t& raw) {
                    if (!t) return;
                    raw.ndims = ggml_n_dims(t);
                    for (int i = 0; i < 4; ++i) raw.ne[i] = t->ne[i];
                    const size_t n = (size_t)ggml_nelements(t);
                    raw.data = (float*)malloc(sizeof(float) * n);
                    if (!raw.data) {
                        raw.ndims = 0;
                        return;
                    }
                    for (size_t i = 0; i < n; ++i) {
                        raw.data[i] = ggml_get_f32_1d(t, (int)i);
                    }
                };

                if (sd_cond.c_crossattn) tensor_to_raw_f32(sd_cond.c_crossattn, cond->c_crossattn);
                if (sd_cond.c_vector) tensor_to_raw_f32(sd_cond.c_vector, cond->c_vector);
                if (sd_cond.c_concat) tensor_to_raw_f32(sd_cond.c_concat, cond->c_concat);

                ggml_free(work_ctx);

            } catch (const std::exception& e) {
                releaseStrings();
                throwJavaException(env, "java/lang/RuntimeException", e.what());
                return nullptr;
            }
        } else {
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

    // We'll return an Object[] = {float[] cross, int[] cross_dims, float[] vec, int[] vec_dims, float[] concat, int[] concat_dims}
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

    // Helper lambda to push arrays
    auto push_tensor = [&](const sd_tensor_raw_t* t, int data_index, int dims_index) {
        if (t == nullptr || t->ndims == 0 || t->data == nullptr) {
            env->SetObjectArrayElement(result, data_index, nullptr);
            env->SetObjectArrayElement(result, dims_index, nullptr);
            return;
        }
        size_t count = 1;
        for (int i = 0; i < t->ndims; ++i) count *= (size_t)t->ne[i];

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
            jint dims[4] = {0,0,0,0};
            for (int i = 0; i < t->ndims && i < 4; ++i) dims[i] = t->ne[i];
            env->SetIntArrayRegion(dimsArr, 0, t->ndims, dims);
            env->SetObjectArrayElement(result, dims_index, dimsArr);
            env->DeleteLocalRef(dimsArr);
        }
    };

    // The condition struct uses sd_tensor_raw_t members; pass by reference to the lambda
    push_tensor(&cond->c_crossattn, 0, 1);
    push_tensor(&cond->c_vector, 2, 3);
    push_tensor(&cond->c_concat, 4, 5);

    // Free native cond buffers
    sd_free_condition(cond);

    return result;
}


// Helper to reconstruct sd_condition_raw_t from Java Object[]
static sd_condition_raw_t* reconstruct_condition(JNIEnv* env, jobjectArray condArr) {
    if (!condArr) return nullptr;

    // Layout: [float[] cross, int[] crossDims, float[] vector, int[] vectorDims, float[] concat, int[] concatDims]
    if (env->GetArrayLength(condArr) < 6) return nullptr;

    auto* cond = (sd_condition_raw_t*)calloc(1, sizeof(sd_condition_raw_t));

    auto extract_tensor = [&](int data_idx, int dims_idx, sd_tensor_raw_t& raw) {
        jfloatArray dataArr = (jfloatArray)env->GetObjectArrayElement(condArr, data_idx);
        jintArray dimsArr = (jintArray)env->GetObjectArrayElement(condArr, dims_idx);

        if (dataArr && dimsArr) {
            jsize dataLen = env->GetArrayLength(dataArr);
            jsize dimsLen = env->GetArrayLength(dimsArr);

            raw.ndims = std::min((int)dimsLen, 4);
            jint* dims = env->GetIntArrayElements(dimsArr, nullptr);
            for(int i=0; i<raw.ndims; ++i) raw.ne[i] = dims[i];
            env->ReleaseIntArrayElements(dimsArr, dims, JNI_ABORT);

            raw.data = (float*)malloc(dataLen * sizeof(float));
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
    extract_tensor(2, 3, cond->c_vector);
    extract_tensor(4, 5, cond->c_concat);

    return cond;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2ImgWithPrecomputedCondition(
    JNIEnv* env, jobject thiz, jlong handlePtr,
    jstring jPrompt, jstring jNegative,
    jint width, jint height,
    jint steps, jfloat cfg, jlong seed,
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

    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    auto releaseStrings = [&]() {
        if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
        if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
    };

    // Reconstruct conditions
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

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.prompt = prompt;
    gen.negative_prompt = negative;
    gen.width = width;
    gen.height = height;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.batch_count = 1;
    gen.easycache.enabled = jEasyCacheEnabled ? true : false;
    gen.easycache.reuse_threshold = (float)jEasyCacheReuseThreshold;
    gen.easycache.start_percent = (float)jEasyCacheStartPercent;
    gen.easycache.end_percent = (float)jEasyCacheEndPercent;

    sd_image_t* out = sd_generate_image_with_precomputed_condition(handle->ctx, &gen, cond, uncond);

    releaseStrings();

    // Cleanup reconstructed conditions
    sd_free_condition(cond);
    if (uncond) sd_free_condition(uncond);

    if (!out || !out[0].data) {
        ALOGE("generate_image failed");
        if (out) free(out);
        return nullptr;
    }

    const size_t byteCount = (size_t)out[0].width * out[0].height * out[0].channel;
    jbyteArray jbytes = env->NewByteArray((jsize)byteCount);
    if (!jbytes) {
        free(out[0].data);
        free(out);
        return nullptr;
    }
    env->SetByteArrayRegion(jbytes, 0, (jsize)byteCount, reinterpret_cast<jbyte*>(out[0].data));

    free(out[0].data);
    free(out);

    return jbytes;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2VidWithPrecomputedCondition(
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
    handle->cancellationRequested.store(false);
    handle->totalFrames = std::max(1, static_cast<int>(videoFrames));
    handle->currentFrame = 0;

    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    auto releaseStrings = [&]() {
        if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
        if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
    };

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    if (cfg > 0.f) sample.guidance.txt_cfg = cfg;

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

    // Map Kotlin enums (DEFAULT=0) to upstream enums (no DEFAULT).
    // Use *_COUNT as a sentinel to request model defaults.
    {
        enum sample_method_t mapped_method;
        if (map_sample_method_from_kotlin_id((int)jSampleMethod, &mapped_method)) {
            gen.sample_params.sample_method = mapped_method;
        } else {
            gen.sample_params.sample_method = SAMPLE_METHOD_COUNT;
        }

        enum scheduler_t mapped_sched;
        if (map_scheduler_from_kotlin_id((int)jScheduler, &mapped_sched)) {
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
            env->GetByteArrayRegion(jInitImage, 0, initSize,
                                    reinterpret_cast<jbyte*>(initImageData.data()));
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

    // Set easycache parameters for both T2V and I2V modes
    gen.easycache.enabled = jEasyCacheEnabled ? true : false;
    gen.easycache.reuse_threshold = (float)jEasyCacheReuseThreshold;
    gen.easycache.start_percent = (float)jEasyCacheStartPercent;
    gen.easycache.end_percent = (float)jEasyCacheEndPercent;

    sd_condition_raw_t* cond_use = reconstruct_condition(env, condArr);
    sd_condition_raw_t* uncond_use = reconstruct_condition(env, uncondArr);

    // Video sampling steps are global for the clip; do not multiply by frame count.
    handle->stepsPerFrame = 0;
    handle->totalSteps = sample.sample_steps > 0 ? sample.sample_steps : 0;

    // Ensure progress callback is wired for cancellation even if Kotlin-side callback is null.
    if (!handle->progressCallbackGlobalRef) {
        sd_set_progress_callback(sd_video_progress_wrapper, handle);
    }

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

    // Free condition buffers built from Java arrays
    if (cond_use) sd_free_condition(cond_use);
    if (uncond_use) sd_free_condition(uncond_use);

    if (!frames || numFrames <= 0) {
        if (frames) {
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
        }
        throwJavaException(env, "java/lang/IllegalStateException", "Video generation failed");
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    jclass byteArrayClass = env->FindClass("[B");
    if (!byteArrayClass) {
        for (int i = 0; i < numFrames; ++i) {
            if (frames[i].data) {
                sd_jni_notify_frame_buffer_freed(frames[i].data);
                free(frames[i].data);
            }
        }
        sd_jni_notify_frame_array_freed(frames);
        free(frames);
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(numFrames, byteArrayClass, nullptr);
    if (!result) {
        for (int i = 0; i < numFrames; ++i) {
            if (frames[i].data) {
                sd_jni_notify_frame_buffer_freed(frames[i].data);
                free(frames[i].data);
            }
        }
        sd_jni_notify_frame_array_freed(frames);
        free(frames);
        throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate video frame array");
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    for (int i = 0; i < numFrames; ++i) {
        if (!frames[i].data) {
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            throwJavaException(env, "java/lang/IllegalStateException", "Missing frame data");
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        const size_t byteCount = static_cast<size_t>(frames[i].width) * frames[i].height * frames[i].channel;
        jbyteArray frameBytes = env->NewByteArray(static_cast<jsize>(byteCount));
        if (!frameBytes) {
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate frame buffer");
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        env->SetByteArrayRegion(frameBytes, 0, static_cast<jsize>(byteCount),
                                 reinterpret_cast<jbyte*>(frames[i].data));
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(frameBytes);
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        env->SetObjectArrayElement(result, i, frameBytes);
        env->DeleteLocalRef(frameBytes);
        sd_jni_notify_frame_buffer_freed(frames[i].data);
        free(frames[i].data);
    }

    sd_jni_notify_frame_array_freed(frames);
    free(frames);
    if (!handle->progressCallbackGlobalRef) {
        sd_set_progress_callback(nullptr, nullptr);
    }
    handle->cancellationRequested.store(false);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeSetProgressCallback(
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

    jclass callbackClass = env->GetObjectClass(progressCallback);
    if (!callbackClass) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Invalid progress callback instance");
        return;
    }
    jmethodID methodId = env->GetMethodID(callbackClass, "onProgress", "(IIIIF)V");
    env->DeleteLocalRef(callbackClass);
    if (!methodId) {
        throwJavaException(env, "java/lang/NoSuchMethodError", "onProgress method not found");
        return;
    }
    jobject globalRef = env->NewGlobalRef(progressCallback);
    if (!globalRef) {
        throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to hold progress callback reference");
        return;
    }

    handle->progressCallbackGlobalRef = globalRef;
    handle->progressMethodID = methodId;
    handle->cancellationRequested.store(false);
    sd_set_progress_callback(sd_video_progress_wrapper, handle);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeCancelGeneration(
        JNIEnv* env, jobject, jlong handlePtr) {
    (void)env;
    if (handlePtr == 0) return;
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    handle->cancellationRequested.store(true);
}
