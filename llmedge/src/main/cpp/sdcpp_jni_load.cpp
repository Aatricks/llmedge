#include <jni.h>

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>

#include "jni_thread_cache.h"
#include "jni_utils.h"

#define GGML_MAX_NAME 128
#include "ggml_backend_probe.h"
#include "sdcpp_jni_shared.h"
#if defined(SD_USE_VULKAN)
#include "ggml-vulkan.h"
#endif
#include "conditioner.hpp"
#include "ggml-backend.h"
#include "model.h"

namespace {
class ScopedEnvVar {
public:
    ScopedEnvVar(const char* key, std::optional<std::string> value) : key_(key), had_original_(false) {
        if (const char* original = std::getenv(key_)) {
            had_original_ = true;
            original_value_ = original;
        }
        if (value.has_value()) {
            setenv(key_, value->c_str(), 1);
        } else {
            unsetenv(key_);
        }
    }

    ~ScopedEnvVar() {
        if (had_original_) {
            setenv(key_, original_value_.c_str(), 1);
        } else {
            unsetenv(key_);
        }
    }

private:
    const char* key_;
    bool had_original_;
    std::string original_value_;
};

static SdHandle* try_create_t5_only_handle(JNIEnv* env, const char* modelPath, bool offloadToCpu, bool useVulkan) {
    if (!modelPath) {
        return nullptr;
    }

    ALOGI("Attempting T5-only context load for sequential prompt conditioning: %s", modelPath);

    ModelLoader model_loader;
    if (!model_loader.init_from_file(modelPath, "text_encoders.t5xxl.transformer.")) {
        ALOGE("Failed to initialize ModelLoader for T5-only context: %s", modelPath);
        return nullptr;
    }
    model_loader.convert_tensors_name();

    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (useVulkan && ggml_backend_vk_get_device_count() > 0) {
        backend = ggml_backend_vk_init(0);
    }
#else
    (void)useVulkan;
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) {
        ALOGE("Unable to initialize backend for T5-only context");
        return nullptr;
    }

    bool is_umt5 = std::string(modelPath).find("umt5") != std::string::npos;
    auto* t5 = new T5CLIPEmbedder(
        backend,
        offloadToCpu,
        model_loader.get_tensor_storage_map(),
        false,
        0,
        is_umt5);
    // alloc_params_buffer() returns void in this fork; load_tensors below is the
    // checkable failure point.
    t5->alloc_params_buffer();

    std::map<std::string, struct ggml_tensor*> tensors;
    t5->get_param_tensors(tensors);
    std::set<std::string> ignore_tensors;
    if (!model_loader.load_tensors(tensors, ignore_tensors, sd_get_num_physical_cores_safe())) {
        ALOGE("Failed to load tensors for T5 context");
        t5->free_params_buffer();
        delete t5;
        ggml_backend_free(backend);
        return nullptr;
    }

    auto* handle = new SdHandle();
    handle->ctx = nullptr;
    handle->t5_ctx = t5;
    handle->backend = backend;
    if (env) {
        env->GetJavaVM(&handle->jvm);
        jni_thread_cache_init(handle->jvm);
    }

    ALOGI("Created T5-only context for sequential prompt conditioning");
    return handle;
}

// Encoder-only handle for FLUX.2 sequential mode (llmedge Lever 1): loads ONLY the Qwen3 text
// encoder (no diffusion transformer), so the precompute phase peaks at the encoder size instead of
// encoder+DiT. Mirrors try_create_t5_only_handle but builds an LLMEmbedder.
static SdHandle* try_create_llm_only_handle(JNIEnv* env, const char* llmPath, bool offloadToCpu, bool useVulkan) {
    if (!llmPath) {
        return nullptr;
    }
    ALOGI("Attempting LLM-only (Qwen3) context load for sequential FLUX.2 conditioning: %s", llmPath);

    ModelLoader model_loader;
    if (!model_loader.init_from_file(llmPath, "text_encoders.llm.")) {
        ALOGE("Failed to initialize ModelLoader for LLM-only context: %s", llmPath);
        return nullptr;
    }
    model_loader.convert_tensors_name();

    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (useVulkan && ggml_backend_vk_get_device_count() > 0) {
        backend = ggml_backend_vk_init(0);
    }
#else
    (void)useVulkan;
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) {
        ALOGE("Unable to initialize backend for LLM-only context");
        return nullptr;
    }

    auto* llm = new LLMEmbedder(
        backend,
        offloadToCpu,
        model_loader.get_tensor_storage_map(),
        VERSION_FLUX2_KLEIN);
    // alloc_params_buffer() returns void in this fork; load_tensors below is the
    // checkable failure point.
    llm->alloc_params_buffer();

    std::map<std::string, struct ggml_tensor*> tensors;
    llm->get_param_tensors(tensors);
    std::set<std::string> ignore_tensors;
    if (!model_loader.load_tensors(tensors, ignore_tensors, sd_get_num_physical_cores_safe())) {
        ALOGE("Failed to load tensors for LLM context");
        llm->free_params_buffer();
        delete llm;
        ggml_backend_free(backend);
        return nullptr;
    }

    auto* handle    = new SdHandle();
    handle->ctx     = nullptr;
    handle->llm_ctx = llm;
    handle->backend = backend;
    if (env) {
        env->GetJavaVM(&handle->jvm);
        jni_thread_cache_init(handle->jvm);
    }
    ALOGI("Created LLM-only (Qwen3) context for sequential FLUX.2 conditioning");
    return handle;
}

void sd_android_log_cb(enum sd_log_level_t level, const char* text, void* data) {
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
} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCheckBindings(JNIEnv*, jclass) {
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeIsOpenClAvailable(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
#ifdef SD_USE_OPENCL
    return llmedge_backend_has_devices("OpenCL", false) ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeGetVulkanDeviceCount(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
#ifdef SD_USE_VULKAN
    return (jint)ggml_backend_vk_get_device_count();
#else
    return 0;
#endif
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeGetVulkanDeviceMemory(JNIEnv* env, jclass clazz, jint deviceIndex) {
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

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeGetVulkanDeviceDescription(JNIEnv* env, jclass clazz, jint deviceIndex) {
    (void)clazz;
#ifdef SD_USE_VULKAN
    char desc[256];
    desc[0] = '\0';
    ggml_backend_vk_get_device_description((int)deviceIndex, desc, sizeof(desc));
    if (desc[0] == '\0') {
        return nullptr;
    }
    return env->NewStringUTF(desc);
#else
    (void)deviceIndex;
    return nullptr;
#endif
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeEstimateModelParamsMemory(JNIEnv* env, jclass clazz, jstring jModelPath, jint deviceIndex) {
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
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeEstimateModelParamsMemoryDetailed(JNIEnv* env, jclass clazz, jstring jModelPath, jint deviceIndex) {
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

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCreate(
    JNIEnv* env, jclass clazz,
        jstring jModelPath,
        jstring jVaePath,
        jstring jT5xxlPath,
        jstring jTaesdPath,
        jstring jDiffusionModelPath,
        jstring jLlmPath,
        jint nThreads,
        jboolean enableOpenCl,
        jboolean useVulkan,
        jboolean offloadToCpu,
        jboolean keepClipOnCpu,
        jboolean keepVaeOnCpu,
        jboolean flashAttn,
        jboolean jvaeDecodeOnly,
        jfloat flowShift,
        jstring jLoraModelDir, jint jLoraApplyMode) {
    (void)clazz;
    const char* modelPath = jModelPath ? env->GetStringUTFChars(jModelPath, nullptr) : nullptr;
    const char* vaePath   = jVaePath   ? env->GetStringUTFChars(jVaePath,   nullptr) : nullptr;
    const char* t5xxlPath = jT5xxlPath ? env->GetStringUTFChars(jT5xxlPath, nullptr) : nullptr;
    const char* taesdPath = jTaesdPath ? env->GetStringUTFChars(jTaesdPath, nullptr) : nullptr;
    const char* loraModelDir = jLoraModelDir ? env->GetStringUTFChars(jLoraModelDir, nullptr) : nullptr;
    const std::string loraModelDirValue = loraModelDir ? loraModelDir : "";

    // FLUX.2 / split-model loading: the diffusion transformer goes in diffusion_model_path
    // (not model_path) and the Qwen3 text encoder in llm_path. Copy into std::string so the
    // values outlive new_sd_ctx without touching the char* release blocks below.
    const char* diffusionModelPathRaw = jDiffusionModelPath ? env->GetStringUTFChars(jDiffusionModelPath, nullptr) : nullptr;
    const std::string diffusionModelPathValue = diffusionModelPathRaw ? diffusionModelPathRaw : "";
    if (jDiffusionModelPath && diffusionModelPathRaw) env->ReleaseStringUTFChars(jDiffusionModelPath, diffusionModelPathRaw);

    const char* llmPathRaw = jLlmPath ? env->GetStringUTFChars(jLlmPath, nullptr) : nullptr;
    const std::string llmPathValue = llmPathRaw ? llmPathRaw : "";
    if (jLlmPath && llmPathRaw) env->ReleaseStringUTFChars(jLlmPath, llmPathRaw);

    sd_set_log_callback(sd_android_log_cb, nullptr);

    ALOGI("Initializing Stable Diffusion with:");
    ALOGI("  modelPath=%s", modelPath ? modelPath : "NULL");
    ALOGI("  vaePath=%s", vaePath ? vaePath : "NULL");
    ALOGI("  t5xxlPath=%s", t5xxlPath ? t5xxlPath : "NULL");
    ALOGI("  taesdPath=%s", taesdPath ? taesdPath : "NULL");
    ALOGI("  diffusionModelPath=%s", diffusionModelPathValue.empty() ? "NULL" : diffusionModelPathValue.c_str());
    ALOGI("  llmPath=%s", llmPathValue.empty() ? "NULL" : llmPathValue.c_str());
    ALOGI("  loraModelDir=%s, loraApplyMode=%d", loraModelDirValue.empty() ? "NULL" : loraModelDirValue.c_str(), static_cast<int>(jLoraApplyMode));
    ALOGI("  enableOpenCl=%s, useVulkan=%s, offloadToCpu=%s, keepClipOnCpu=%s, keepVaeOnCpu=%s, flashAttn=%s, vaeDecodeOnly=%s",
          enableOpenCl ? "true" : "false",
          useVulkan ? "true" : "false",
          offloadToCpu ? "true" : "false",
          keepClipOnCpu ? "true" : "false",
          keepVaeOnCpu ? "true" : "false",
          flashAttn ? "true" : "false",
          jvaeDecodeOnly ? "true" : "false");

    std::optional<std::string> selectedVulkanDevice;
#ifdef SD_USE_VULKAN
    if (useVulkan == JNI_TRUE) {
        const int device_count = ggml_backend_vk_get_device_count();
        if (device_count > 0) {
            int best = 0;
            size_t best_total = 0;
            for (int i = 0; i < device_count; ++i) {
                size_t free_mem = 0, total_mem = 0;
                ggml_backend_vk_get_device_memory(i, &free_mem, &total_mem);
                char desc[256];
                desc[0] = '\0';
                ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
                ALOGI("Vulkan device %d: %s free=%zu total=%zu", i, desc[0] ? desc : "(unknown)", free_mem, total_mem);
                if (total_mem > best_total) {
                    best_total = total_mem;
                    best = i;
                }
            }
            ALOGI("Selecting Vulkan device %d", best);
            selectedVulkanDevice = std::to_string(best);
        } else {
            ALOGW("Vulkan requested but ggml reported 0 Vulkan devices");
        }
    }
#endif

    sd_ctx_params_t p{};
    sd_ctx_params_init(&p);
    p.model_path = modelPath ? modelPath : "";
    p.vae_path = vaePath ? vaePath : "";
    p.t5xxl_path = t5xxlPath;
    p.taesd_path = taesdPath ? taesdPath : "";
    // Split-model components (FLUX.2 Klein etc.). Empty string == not provided.
    p.diffusion_model_path = diffusionModelPathValue.c_str();
    p.llm_path = llmPathValue.c_str();
    p.free_params_immediately = true;
    p.n_threads = nThreads > 0 ? nThreads : sd_get_num_physical_cores_safe();
    p.offload_params_to_cpu = offloadToCpu;
    p.keep_clip_on_cpu = keepClipOnCpu;
    p.keep_vae_on_cpu = keepVaeOnCpu;
    p.diffusion_flash_attn = flashAttn;
    p.vae_decode_only = jvaeDecodeOnly;
    p.lora_apply_mode = static_cast<enum lora_apply_mode_t>(jLoraApplyMode);

    bool isLlmOnly = !llmPathValue.empty() && diffusionModelPathValue.empty() && (!modelPath || modelPath[0] == '\0');
    bool isT5OnlyRequest = modelPath && modelPath[0] != '\0' &&
                           !vaePath && !t5xxlPath && !taesdPath &&
                           diffusionModelPathValue.empty() && llmPathValue.empty() &&
                           (std::string(modelPath).find("t5") != std::string::npos ||
                            std::string(modelPath).find("encoder") != std::string::npos ||
                            std::string(modelPath).find("clip") != std::string::npos);

    sd_ctx_t* ctx = nullptr;
    if (!isLlmOnly) {
        // Shared with the whisper loader: env mutation must be process-globally
        // serialized (concurrent setenv/getenv across threads is UB).
        std::lock_guard<std::mutex> lock(llmedge_process_env_mutex());
        ScopedEnvVar disableVulkan("GGML_DISABLE_VULKAN", useVulkan == JNI_TRUE ? std::nullopt : std::optional<std::string>("1"));
        ScopedEnvVar disableOpenCl("GGML_DISABLE_OPENCL", enableOpenCl == JNI_TRUE ? std::nullopt : std::optional<std::string>("1"));
        ScopedEnvVar vulkanDevice("SD_VK_DEVICE", selectedVulkanDevice);
        ctx = new_sd_ctx(&p);
    }

    if (!ctx) {
        if (isLlmOnly) {
            SdHandle* llmOnlyHandle = try_create_llm_only_handle(env, llmPathValue.c_str(), offloadToCpu == JNI_TRUE, useVulkan == JNI_TRUE);
            if (llmOnlyHandle) {
                if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
                if (jLoraModelDir) env->ReleaseStringUTFChars(jLoraModelDir, loraModelDir);
                return reinterpret_cast<jlong>(llmOnlyHandle);
            }
        }
        if (isT5OnlyRequest) {
            SdHandle* t5OnlyHandle = try_create_t5_only_handle(env, modelPath, offloadToCpu == JNI_TRUE, useVulkan == JNI_TRUE);
            if (t5OnlyHandle) {
                if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
                if (jLoraModelDir) env->ReleaseStringUTFChars(jLoraModelDir, loraModelDir);
                return reinterpret_cast<jlong>(t5OnlyHandle);
            }
        }

        ALOGE("Failed to create sd_ctx");
        if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
        if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
        if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
        if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
        if (jLoraModelDir) env->ReleaseStringUTFChars(jLoraModelDir, loraModelDir);
        return 0;
    }

    if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
    if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
    if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
    if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
    if (jLoraModelDir) env->ReleaseStringUTFChars(jLoraModelDir, loraModelDir);

    auto* handle = new SdHandle();
    handle->ctx = ctx;
    handle->flowShift = flowShift;
    handle->loraModelDir = loraModelDirValue;
    if (env) {
        env->GetJavaVM(&handle->jvm);
        jni_thread_cache_init(handle->jvm);
    }
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(JNIEnv* env, jobject, jlong handlePtr) {
    if (handlePtr == 0) return;
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    clearProgressCallback(env, handle);
    // Do NOT clear the process-global sd progress callback here: it may belong to a
    // different live handle (FLUX.2 split mode holds encoder + diffusion handles).
    // Generation paths set and clear the global callback around themselves.
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
    if (handle->llm_ctx) {
        auto* llm = static_cast<LLMEmbedder*>(handle->llm_ctx);
        llm->free_params_buffer();
        delete llm;
        handle->llm_ctx = nullptr;
    }
    if (handle->backend) {
        ggml_backend_free(static_cast<ggml_backend_t>(handle->backend));
        handle->backend = nullptr;
    }
    delete handle;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeIsEasyCacheSupported(JNIEnv* env, jobject, jlong handlePtr) {
    (void)env;
    if (handlePtr == 0) {
        return JNI_FALSE;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    if (!handle->ctx) {
        return JNI_FALSE;
    }
    return sd_get_default_sample_method(handle->ctx) == EULER_SAMPLE_METHOD ? JNI_TRUE : JNI_FALSE;
}
