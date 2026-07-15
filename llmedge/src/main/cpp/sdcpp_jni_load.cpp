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
// conditioner.hpp drags in the full header-only sd.cpp model graph code, which the
// host-native test harness cannot link (it stubs libstable-diffusion instead).
#ifndef SD_JNI_TESTING
#include "conditioning/conditioner.hpp"
#endif
#include "core/ggml_extend_backend.h"
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

#ifndef SD_JNI_TESTING
class StreamingModelManager : public ModelManager {
public:
    void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override {
        ModelManager::release_compute_backend_params(tensors);
        ModelManager::release_params_backend_params(tensors);
    }
};
#endif

static SdHandle* try_create_t5_only_handle(
        JNIEnv* env,
        const char* modelPath,
        bool offloadToCpu,
        bool useVulkan,
        bool miniT2iConditionerOnly,
        bool chromaT5ConditionerOnly) {
#ifdef SD_JNI_TESTING
    (void)env; (void)modelPath; (void)offloadToCpu; (void)useVulkan; (void)miniT2iConditionerOnly; (void)chromaT5ConditionerOnly;
    return nullptr;
#else
    if (!modelPath) {
        return nullptr;
    }

    ALOGI("Attempting %s-only context load for sequential prompt conditioning: %s",
          miniT2iConditionerOnly ? "MiniT2I" : (chromaT5ConditionerOnly ? "Chroma" : "T5"), modelPath);

    auto model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(sd_get_num_physical_cores_safe());
    model_manager->set_enable_mmap(true);
    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file(modelPath, "text_encoders.t5xxl.transformer.")) {
        ALOGE("Failed to initialize ModelLoader for T5-only context: %s", modelPath);
        return nullptr;
    }
    model_loader.convert_tensors_name();
    model_loader.process_model_files(true, false);

    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (useVulkan && ggml_backend_vk_get_device_count() > 0) {
        backend = ggml_backend_vk_init(0);
    }
#else
    (void)useVulkan;
#endif
    if (!backend) {
        backend = sd_backend_cpu_init();
    }
    if (!backend) {
        ALOGE("Unable to initialize backend for T5-only context");
        return nullptr;
    }

    ggml_backend_t params_backend = (offloadToCpu && !sd_backend_is_cpu(backend)) ? sd_backend_cpu_init() : backend;
    Conditioner* conditioner = nullptr;
    if (miniT2iConditionerOnly) {
        conditioner = new MiniT2IConditioner(
            backend,
            model_manager->loader().get_tensor_storage_map(),
            model_manager);
    } else if (chromaT5ConditionerOnly) {
        conditioner = new T5CLIPEmbedder(
            backend,
            model_manager->loader().get_tensor_storage_map(),
            false, // use_mask
            1,     // mask_pad
            false, // is_umt5
            model_manager);
    } else {
        const bool is_umt5 = std::string(modelPath).find("umt5") != std::string::npos;
        conditioner = new T5CLIPEmbedder(
            backend,
            model_manager->loader().get_tensor_storage_map(),
            false,
            0,
            is_umt5,
            model_manager);
    }
    std::string param_desc = miniT2iConditionerOnly ? "MiniT2I encoder" : (chromaT5ConditionerOnly ? "Chroma encoder" : "T5 encoder");
    if (!model_manager->register_runner_params(
            param_desc,
            *conditioner,
            ModelManager::ResidencyMode::ParamBackend,
            backend,
            params_backend) ||
        !model_manager->validate_registered_tensors()) {
        ALOGE("Failed to register tensors for %s", param_desc.c_str());
        model_manager->unregister_param_tensors(param_desc);
        delete conditioner;
        model_manager.reset();
        if (params_backend != backend) ggml_backend_free(params_backend);
        ggml_backend_free(backend);
        return nullptr;
    }

    auto* handle = new SdHandle();
    handle->ctx = nullptr;
    if (miniT2iConditionerOnly) {
        handle->minit2i_cond_ctx = conditioner;
    } else {
        handle->t5_ctx = conditioner;
    }
    handle->backend = backend;
    handle->params_backend = params_backend == backend ? nullptr : params_backend;
    handle->model_manager = std::move(model_manager);
    handle->param_desc = param_desc;
    if (env) {
        env->GetJavaVM(&handle->jvm);
        jni_thread_cache_init(handle->jvm);
    }

    ALOGI("Created context for %s for sequential prompt conditioning", param_desc.c_str());
    return handle;
#endif
}

static SdHandle* try_create_sd3_encoder_only_handle(JNIEnv* env, const char* clipLPath, const char* clipGPath, const char* t5xxlPath, bool offloadToCpu, bool useVulkan) {
#ifdef SD_JNI_TESTING
    (void)env; (void)clipLPath; (void)clipGPath; (void)t5xxlPath; (void)offloadToCpu; (void)useVulkan;
    return nullptr;
#else
    if (!clipLPath && !clipGPath && !t5xxlPath) {
        return nullptr;
    }
    ALOGI("Attempting SD3 encoder-only context load for sequential conditioning: clipL=%s, clipG=%s, t5xxl=%s",
          clipLPath ? clipLPath : "NULL", clipGPath ? clipGPath : "NULL", t5xxlPath ? t5xxlPath : "NULL");

    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (useVulkan && ggml_backend_vk_get_device_count() > 0) {
        backend = ggml_backend_vk_init(0);
    }
#else
    (void)useVulkan;
#endif
    if (!backend) {
        backend = sd_backend_cpu_init();
    }
    if (!backend) {
        ALOGE("Unable to initialize backend for SD3 encoder-only context");
        return nullptr;
    }

    const bool is_t5_only = (t5xxlPath && t5xxlPath[0] != '\0') &&
                            (!clipLPath || clipLPath[0] == '\0') &&
                            (!clipGPath || clipGPath[0] == '\0');
    const bool is_non_cpu = !sd_backend_is_cpu(backend);
    const bool use_streaming = is_t5_only && is_non_cpu;

    std::shared_ptr<ModelManager> model_manager;
    if (use_streaming) {
        model_manager = std::make_shared<StreamingModelManager>();
    } else {
        model_manager = std::make_shared<ModelManager>();
    }

    model_manager->set_n_threads(sd_get_num_physical_cores_safe());
    model_manager->set_enable_mmap(true);
    ModelLoader& model_loader = model_manager->loader();
    if (clipLPath && clipLPath[0] != '\0') {
        if (!model_loader.init_from_file(clipLPath, "text_encoders.clip_l.transformer.")) {
            ALOGE("Failed to initialize ModelLoader for CLIP-L: %s", clipLPath);
            ggml_backend_free(backend);
            return nullptr;
        }
    }
    if (clipGPath && clipGPath[0] != '\0') {
        if (!model_loader.init_from_file(clipGPath, "text_encoders.clip_g.transformer.")) {
            ALOGE("Failed to initialize ModelLoader for CLIP-G: %s", clipGPath);
            ggml_backend_free(backend);
            return nullptr;
        }
    }
    if (t5xxlPath && t5xxlPath[0] != '\0') {
        if (!model_loader.init_from_file(t5xxlPath, "text_encoders.t5xxl.transformer.")) {
            ALOGE("Failed to initialize ModelLoader for T5XXL: %s", t5xxlPath);
            ggml_backend_free(backend);
            return nullptr;
        }
    }
    model_loader.convert_tensors_name();
    model_loader.process_model_files(true, false);

    ggml_backend_t params_backend;
    if (use_streaming) {
        params_backend = backend;
    } else {
        params_backend = (offloadToCpu && !sd_backend_is_cpu(backend)) ? sd_backend_cpu_init() : backend;
    }

    auto* sd3 = new SD3CLIPEmbedder(
        backend,
        model_manager->loader().get_tensor_storage_map(),
        model_manager
    );

    if (use_streaming) {
        sd3->set_stream_layers_enabled(true);
        sd3->set_max_graph_vram_bytes(1024ULL * 1024ULL * 1024ULL);
        ALOGI("SD3 T5-only encoder layer streaming enabled with budget: 1024 MiB");
    }

    bool reg_ok = false;
    if (use_streaming) {
        std::map<std::string, ggml_tensor*> tensors;
        sd3->get_param_tensors(tensors);
        reg_ok = model_manager->register_param_tensors(
            "SD3 encoder",
            std::move(tensors),
            ModelManager::ResidencyMode::Disk,
            backend,
            params_backend,
            nullptr, // registered_tensor_size
            false,   // allow_split_buffer
            true     // params_follow_compute_backend
        );
    } else {
        reg_ok = model_manager->register_runner_params(
            "SD3 encoder",
            *sd3,
            ModelManager::ResidencyMode::ParamBackend,
            backend,
            params_backend
        );
    }

    if (!reg_ok || !model_manager->validate_registered_tensors()) {
        ALOGE("Failed to register tensors for SD3 context");
        delete sd3;
        model_manager.reset();
        if (params_backend != backend) ggml_backend_free(params_backend);
        ggml_backend_free(backend);
        return nullptr;
    }

    auto* handle = new SdHandle();
    handle->ctx = nullptr;
    handle->sd3_cond_ctx = sd3;
    handle->backend = backend;
    handle->params_backend = params_backend == backend ? nullptr : params_backend;
    handle->model_manager = std::move(model_manager);
    if (env) {
        env->GetJavaVM(&handle->jvm);
        jni_thread_cache_init(handle->jvm);
    }
    ALOGI("Created SD3 encoder-only context for sequential conditioning");
    return handle;
#endif
}

// Encoder-only handle for FLUX.2 sequential mode (llmedge Lever 1): loads ONLY the Qwen3 text
// encoder (no diffusion transformer), so the precompute phase peaks at the encoder size instead of
// encoder+DiT. Mirrors try_create_t5_only_handle but builds an LLMEmbedder.
static SdHandle* try_create_llm_only_handle(JNIEnv* env, const char* llmPath, bool offloadToCpu, bool useVulkan) {
#ifdef SD_JNI_TESTING
    (void)env; (void)llmPath; (void)offloadToCpu; (void)useVulkan;
    return nullptr;
#else
    if (!llmPath) {
        return nullptr;
    }
    ALOGI("Attempting LLM-only (Qwen3) context load for sequential FLUX.2 conditioning: %s", llmPath);

    auto model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(sd_get_num_physical_cores_safe());
    model_manager->set_enable_mmap(true);
    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file(llmPath, "text_encoders.llm.")) {
        ALOGE("Failed to initialize ModelLoader for LLM-only context: %s", llmPath);
        return nullptr;
    }
    model_loader.convert_tensors_name();
    model_loader.process_model_files(true, false);

    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (useVulkan && ggml_backend_vk_get_device_count() > 0) {
        backend = ggml_backend_vk_init(0);
    }
#else
    (void)useVulkan;
#endif
    if (!backend) {
        backend = sd_backend_cpu_init();
    }
    if (!backend) {
        ALOGE("Unable to initialize backend for LLM-only context");
        return nullptr;
    }

    ggml_backend_t params_backend = offloadToCpu ? sd_backend_cpu_init() : backend;
    auto* llm = new LLMEmbedder(
        backend,
        model_manager->loader().get_tensor_storage_map(),
        VERSION_FLUX2_KLEIN,
        "",
        false,
        model_manager);
    if (!model_manager->register_runner_params(
            "LLM encoder",
            *llm,
            ModelManager::ResidencyMode::ParamBackend,
            backend,
            params_backend) ||
        !model_manager->validate_registered_tensors()) {
        ALOGE("Failed to register tensors for LLM context");
        delete llm;
        model_manager.reset();
        if (params_backend != backend) ggml_backend_free(params_backend);
        ggml_backend_free(backend);
        return nullptr;
    }

    auto* handle    = new SdHandle();
    handle->ctx     = nullptr;
    handle->llm_ctx = llm;
    handle->backend = backend;
    handle->params_backend = params_backend == backend ? nullptr : params_backend;
    handle->model_manager = std::move(model_manager);
    if (env) {
        env->GetJavaVM(&handle->jvm);
        jni_thread_cache_init(handle->jvm);
    }
    ALOGI("Created LLM-only (Qwen3) context for sequential FLUX.2 conditioning");
    return handle;
#endif
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
    // Probing must never kill the process: honor the registry kill switch and
    // catch loader failures (a Vulkan loader with no working driver throws
    // vk::IncompatibleDriverError out of instance creation).
    if (std::getenv("GGML_DISABLE_VULKAN") != nullptr) {
        return 0;
    }
    try {
        return (jint)ggml_backend_vk_get_device_count();
    } catch (const std::exception& e) {
        ALOGW("Vulkan device probe failed, falling back to 0 devices: %s", e.what());
        return 0;
    } catch (...) {
        ALOGW("Vulkan device probe failed with a non-standard exception; falling back to 0 devices");
        return 0;
    }
#else
    return 0;
#endif
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeGetVulkanDeviceMemory(JNIEnv* env, jclass clazz, jint deviceIndex) {
    (void)clazz;
#ifdef SD_USE_VULKAN
    size_t free_mem = 0, total_mem = 0;
    try {
        ggml_backend_vk_get_device_memory((int)deviceIndex, &free_mem, &total_mem);
    } catch (const std::exception& e) {
        ALOGW("Vulkan memory probe failed: %s", e.what());
        free_mem = 0;
        total_mem = 0;
    }
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
    try {
        ggml_backend_vk_get_device_description((int)deviceIndex, desc, sizeof(desc));
    } catch (const std::exception& e) {
        ALOGW("Vulkan description probe failed: %s", e.what());
    }
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
        jstring jClipLPath,
        jstring jClipGPath,
        jstring jClipVisionPath,
        jstring jLlmVisionPath,
        jstring jHighNoiseDiffusionModelPath,
        jstring jEmbeddingsConnectorsPath,
        jstring jAudioVaePath,
        jstring jControlNetPath,
        jstring jPhotoMakerPath,
        jint nThreads,
        jboolean enableOpenCl,
        jboolean useVulkan,
        jboolean offloadToCpu,
        jboolean keepClipOnCpu,
        jboolean keepVaeOnCpu,
        jboolean flashAttn,
        jboolean jvaeDecodeOnly,
        jfloat flowShift,
        jstring jLoraModelDir, jint jLoraApplyMode, jboolean jMiniT2iConditionerOnly, jboolean jChromaT5ConditionerOnly,
        jstring jWeightType,
        jstring jTensorTypeRules) {
    (void)clazz;
    const char* modelPath = jModelPath ? env->GetStringUTFChars(jModelPath, nullptr) : nullptr;
    const char* vaePath   = jVaePath   ? env->GetStringUTFChars(jVaePath,   nullptr) : nullptr;
    const char* t5xxlPath = jT5xxlPath ? env->GetStringUTFChars(jT5xxlPath, nullptr) : nullptr;
    const char* taesdPath = jTaesdPath ? env->GetStringUTFChars(jTaesdPath, nullptr) : nullptr;
    const char* loraModelDir = jLoraModelDir ? env->GetStringUTFChars(jLoraModelDir, nullptr) : nullptr;
    const std::string loraModelDirValue = loraModelDir ? loraModelDir : "";

    const char* weightTypeRaw = jWeightType ? env->GetStringUTFChars(jWeightType, nullptr) : nullptr;
    const char* tensorTypeRulesRaw = jTensorTypeRules ? env->GetStringUTFChars(jTensorTypeRules, nullptr) : nullptr;
    const std::string tensorTypeRulesValue = tensorTypeRulesRaw ? tensorTypeRulesRaw : "";
    if (jTensorTypeRules && tensorTypeRulesRaw) env->ReleaseStringUTFChars(jTensorTypeRules, tensorTypeRulesRaw);

    // FLUX.2 / split-model loading: the diffusion transformer goes in diffusion_model_path
    // (not model_path) and the Qwen3 text encoder in llm_path. Copy into std::string so the
    // values outlive new_sd_ctx without touching the char* release blocks below.
    const char* diffusionModelPathRaw = jDiffusionModelPath ? env->GetStringUTFChars(jDiffusionModelPath, nullptr) : nullptr;
    const std::string diffusionModelPathValue = diffusionModelPathRaw ? diffusionModelPathRaw : "";
    if (jDiffusionModelPath && diffusionModelPathRaw) env->ReleaseStringUTFChars(jDiffusionModelPath, diffusionModelPathRaw);

    const char* llmPathRaw = jLlmPath ? env->GetStringUTFChars(jLlmPath, nullptr) : nullptr;
    const std::string llmPathValue = llmPathRaw ? llmPathRaw : "";
    if (jLlmPath && llmPathRaw) env->ReleaseStringUTFChars(jLlmPath, llmPathRaw);

    const char* clipLPathRaw = jClipLPath ? env->GetStringUTFChars(jClipLPath, nullptr) : nullptr;
    const std::string clipLPathValue = clipLPathRaw ? clipLPathRaw : "";
    if (jClipLPath && clipLPathRaw) env->ReleaseStringUTFChars(jClipLPath, clipLPathRaw);

    const char* clipGPathRaw = jClipGPath ? env->GetStringUTFChars(jClipGPath, nullptr) : nullptr;
    const std::string clipGPathValue = clipGPathRaw ? clipGPathRaw : "";
    if (jClipGPath && clipGPathRaw) env->ReleaseStringUTFChars(jClipGPath, clipGPathRaw);

    const char* clipVisionPathRaw = jClipVisionPath ? env->GetStringUTFChars(jClipVisionPath, nullptr) : nullptr;
    const std::string clipVisionPathValue = clipVisionPathRaw ? clipVisionPathRaw : "";
    if (jClipVisionPath && clipVisionPathRaw) env->ReleaseStringUTFChars(jClipVisionPath, clipVisionPathRaw);

    const char* llmVisionPathRaw = jLlmVisionPath ? env->GetStringUTFChars(jLlmVisionPath, nullptr) : nullptr;
    const std::string llmVisionPathValue = llmVisionPathRaw ? llmVisionPathRaw : "";
    if (jLlmVisionPath && llmVisionPathRaw) env->ReleaseStringUTFChars(jLlmVisionPath, llmVisionPathRaw);

    const char* highNoiseDiffusionModelPathRaw = jHighNoiseDiffusionModelPath ? env->GetStringUTFChars(jHighNoiseDiffusionModelPath, nullptr) : nullptr;
    const std::string highNoiseDiffusionModelPathValue = highNoiseDiffusionModelPathRaw ? highNoiseDiffusionModelPathRaw : "";
    if (jHighNoiseDiffusionModelPath && highNoiseDiffusionModelPathRaw) env->ReleaseStringUTFChars(jHighNoiseDiffusionModelPath, highNoiseDiffusionModelPathRaw);

    const char* embeddingsConnectorsPathRaw = jEmbeddingsConnectorsPath ? env->GetStringUTFChars(jEmbeddingsConnectorsPath, nullptr) : nullptr;
    const std::string embeddingsConnectorsPathValue = embeddingsConnectorsPathRaw ? embeddingsConnectorsPathRaw : "";
    if (jEmbeddingsConnectorsPath && embeddingsConnectorsPathRaw) env->ReleaseStringUTFChars(jEmbeddingsConnectorsPath, embeddingsConnectorsPathRaw);

    const char* audioVaePathRaw = jAudioVaePath ? env->GetStringUTFChars(jAudioVaePath, nullptr) : nullptr;
    const std::string audioVaePathValue = audioVaePathRaw ? audioVaePathRaw : "";
    if (jAudioVaePath && audioVaePathRaw) env->ReleaseStringUTFChars(jAudioVaePath, audioVaePathRaw);

    const char* controlNetPathRaw = jControlNetPath ? env->GetStringUTFChars(jControlNetPath, nullptr) : nullptr;
    const std::string controlNetPathValue = controlNetPathRaw ? controlNetPathRaw : "";
    if (jControlNetPath && controlNetPathRaw) env->ReleaseStringUTFChars(jControlNetPath, controlNetPathRaw);

    const char* photoMakerPathRaw = jPhotoMakerPath ? env->GetStringUTFChars(jPhotoMakerPath, nullptr) : nullptr;
    const std::string photoMakerPathValue = photoMakerPathRaw ? photoMakerPathRaw : "";
    if (jPhotoMakerPath && photoMakerPathRaw) env->ReleaseStringUTFChars(jPhotoMakerPath, photoMakerPathRaw);

    sd_set_log_callback(sd_android_log_cb, nullptr);

    ALOGI("Initializing Stable Diffusion with:");
    ALOGI("  modelPath=%s", modelPath ? modelPath : "NULL");
    ALOGI("  vaePath=%s", vaePath ? vaePath : "NULL");
    ALOGI("  t5xxlPath=%s", t5xxlPath ? t5xxlPath : "NULL");
    ALOGI("  taesdPath=%s", taesdPath ? taesdPath : "NULL");
    ALOGI("  diffusionModelPath=%s", diffusionModelPathValue.empty() ? "NULL" : diffusionModelPathValue.c_str());
    ALOGI("  llmPath=%s", llmPathValue.empty() ? "NULL" : llmPathValue.c_str());
    ALOGI("  clipLPath=%s", clipLPathValue.empty() ? "NULL" : clipLPathValue.c_str());
    ALOGI("  clipGPath=%s", clipGPathValue.empty() ? "NULL" : clipGPathValue.c_str());
    ALOGI("  clipVisionPath=%s", clipVisionPathValue.empty() ? "NULL" : clipVisionPathValue.c_str());
    ALOGI("  llmVisionPath=%s", llmVisionPathValue.empty() ? "NULL" : llmVisionPathValue.c_str());
    ALOGI("  highNoiseDiffusionModelPath=%s", highNoiseDiffusionModelPathValue.empty() ? "NULL" : highNoiseDiffusionModelPathValue.c_str());
    ALOGI("  embeddingsConnectorsPath=%s", embeddingsConnectorsPathValue.empty() ? "NULL" : embeddingsConnectorsPathValue.c_str());
    ALOGI("  audioVaePath=%s", audioVaePathValue.empty() ? "NULL" : audioVaePathValue.c_str());
    ALOGI("  controlNetPath=%s", controlNetPathValue.empty() ? "NULL" : controlNetPathValue.c_str());
    ALOGI("  photoMakerPath=%s", photoMakerPathValue.empty() ? "NULL" : photoMakerPathValue.c_str());
    ALOGI("  loraModelDir=%s, loraApplyMode=%d", loraModelDirValue.empty() ? "NULL" : loraModelDirValue.c_str(), static_cast<int>(jLoraApplyMode));
    ALOGI("  weightType=%s, tensorTypeRules=%s", weightTypeRaw && weightTypeRaw[0] != '\0' ? weightTypeRaw : "DEFAULT", tensorTypeRulesValue.empty() ? "NULL" : tensorTypeRulesValue.c_str());
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
    p.enable_mmap = true;
    p.model_path = modelPath ? modelPath : "";
    p.vae_path = vaePath ? vaePath : "";
    p.t5xxl_path = t5xxlPath;
    p.taesd_path = taesdPath ? taesdPath : "";
    // Split-model components (FLUX.2 Klein etc.). Empty string == not provided.
    p.diffusion_model_path = diffusionModelPathValue.c_str();
    p.llm_path = llmPathValue.c_str();
    p.clip_l_path = clipLPathValue.c_str();
    p.clip_g_path = clipGPathValue.c_str();
    p.clip_vision_path = clipVisionPathValue.c_str();
    p.llm_vision_path = llmVisionPathValue.c_str();
    p.high_noise_diffusion_model_path = highNoiseDiffusionModelPathValue.c_str();
    p.embeddings_connectors_path = embeddingsConnectorsPathValue.c_str();
    p.audio_vae_path = audioVaePathValue.c_str();
    p.control_net_path = controlNetPathValue.c_str();
    p.photo_maker_path = photoMakerPathValue.c_str();
    p.n_threads = nThreads > 0 ? nThreads : sd_get_num_physical_cores_safe();
    p.diffusion_flash_attn = flashAttn;
    p.lora_apply_mode = static_cast<enum lora_apply_mode_t>(jLoraApplyMode);

    if (weightTypeRaw && weightTypeRaw[0] != '\0') {
        enum sd_type_t parsedType = str_to_sd_type(weightTypeRaw);
        if (parsedType == SD_TYPE_COUNT) {
            ALOGW("Invalid weightType '%s' requested, keeping default", weightTypeRaw);
        } else {
            p.wtype = parsedType;
        }
    }
    if (jWeightType && weightTypeRaw) {
        env->ReleaseStringUTFChars(jWeightType, weightTypeRaw);
    }
    p.tensor_type_rules = tensorTypeRulesValue.empty() ? nullptr : tensorTypeRulesValue.c_str();

    std::string backendSpec;
    if (keepClipOnCpu == JNI_TRUE) backendSpec = "te=cpu";
    if (keepVaeOnCpu == JNI_TRUE) {
        if (!backendSpec.empty()) backendSpec += ",";
        backendSpec += "vae=cpu";
    }
    std::string paramsBackendSpec = offloadToCpu == JNI_TRUE ? "*=cpu" : "";
    p.backend = backendSpec.c_str();
    p.params_backend = paramsBackendSpec.c_str();

    bool hasExtraComponentPaths = !clipLPathValue.empty() ||
                                  !clipGPathValue.empty() ||
                                  !clipVisionPathValue.empty() ||
                                  !llmVisionPathValue.empty() ||
                                  !highNoiseDiffusionModelPathValue.empty() ||
                                  !embeddingsConnectorsPathValue.empty() ||
                                  !audioVaePathValue.empty() ||
                                  !controlNetPathValue.empty() ||
                                  !photoMakerPathValue.empty();

    bool isChromaT5ConditionerOnly = jChromaT5ConditionerOnly == JNI_TRUE;

    bool isSd3EncoderOnly = !isChromaT5ConditionerOnly &&
                            (diffusionModelPathValue.empty()) &&
                            (modelPath == nullptr || modelPath[0] == '\0') &&
                            (vaePath == nullptr || vaePath[0] == '\0') &&
                            (llmPathValue.empty()) &&
                            (
                              ((t5xxlPath == nullptr || t5xxlPath[0] == '\0') && (!clipLPathValue.empty()) && (!clipGPathValue.empty())) ||
                              ((t5xxlPath != nullptr && t5xxlPath[0] != '\0') && (clipLPathValue.empty()) && (clipGPathValue.empty()))
                            );

    bool isLlmOnly = !llmPathValue.empty() && diffusionModelPathValue.empty() && (!modelPath || modelPath[0] == '\0') && !hasExtraComponentPaths;
    bool isT5OnlyRequest = modelPath && modelPath[0] != '\0' &&
                           !vaePath && !t5xxlPath && !taesdPath &&
                           diffusionModelPathValue.empty() && llmPathValue.empty() &&
                           !hasExtraComponentPaths &&
                           (std::string(modelPath).find("t5") != std::string::npos ||
                            std::string(modelPath).find("encoder") != std::string::npos);
    bool isMiniT2iConditionerOnly = jMiniT2iConditionerOnly == JNI_TRUE;

    sd_ctx_t* ctx = nullptr;
    if (!isLlmOnly && !isSd3EncoderOnly && !isMiniT2iConditionerOnly && !isChromaT5ConditionerOnly) {
        // Shared with the whisper loader: env mutation must be process-globally
        // serialized (concurrent setenv/getenv across threads is UB).
        std::lock_guard<std::mutex> lock(llmedge_process_env_mutex());
        ScopedEnvVar disableVulkan("GGML_DISABLE_VULKAN", useVulkan == JNI_TRUE ? std::nullopt : std::optional<std::string>("1"));
        ScopedEnvVar disableOpenCl("GGML_DISABLE_OPENCL", enableOpenCl == JNI_TRUE ? std::nullopt : std::optional<std::string>("1"));
        ScopedEnvVar vulkanDevice("SD_VK_DEVICE", selectedVulkanDevice);
        ctx = new_sd_ctx(&p);
    }

    if (!ctx) {
        if (isChromaT5ConditionerOnly) {
            SdHandle* chromaHandle = try_create_t5_only_handle(
                env, t5xxlPath, offloadToCpu == JNI_TRUE, useVulkan == JNI_TRUE, false, true);
            if (chromaHandle) {
                if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
                if (jLoraModelDir) env->ReleaseStringUTFChars(jLoraModelDir, loraModelDir);
                return reinterpret_cast<jlong>(chromaHandle);
            }
        }
        if (isSd3EncoderOnly) {
            SdHandle* sd3EncoderHandle = try_create_sd3_encoder_only_handle(env, clipLPathValue.c_str(), clipGPathValue.c_str(), t5xxlPath, offloadToCpu == JNI_TRUE, useVulkan == JNI_TRUE);
            if (sd3EncoderHandle) {
                if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
                if (jLoraModelDir) env->ReleaseStringUTFChars(jLoraModelDir, loraModelDir);
                return reinterpret_cast<jlong>(sd3EncoderHandle);
            }
        }
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
        if (isMiniT2iConditionerOnly) {
            SdHandle* miniT2iHandle = try_create_t5_only_handle(
                env, modelPath, offloadToCpu == JNI_TRUE, useVulkan == JNI_TRUE, true, false);
            if (miniT2iHandle) {
                if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                if (jTaesdPath) env->ReleaseStringUTFChars(jTaesdPath, taesdPath);
                if (jLoraModelDir) env->ReleaseStringUTFChars(jLoraModelDir, loraModelDir);
                return reinterpret_cast<jlong>(miniT2iHandle);
            }
        }
        if (isT5OnlyRequest) {
            SdHandle* t5OnlyHandle = try_create_t5_only_handle(
                env, modelPath, offloadToCpu == JNI_TRUE, useVulkan == JNI_TRUE, false, false);
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
#ifndef SD_JNI_TESTING
    if (handle->model_manager && !handle->param_desc.empty()) {
        if (!handle->model_manager->unregister_param_tensors(handle->param_desc)) {
            ALOGE("Failed to unregister parameter tensors for %s", handle->param_desc.c_str());
        }
    }
    if (handle->t5_ctx) {
        auto* t5 = static_cast<T5CLIPEmbedder*>(handle->t5_ctx);
        delete t5;
        handle->t5_ctx = nullptr;
    }
    if (handle->minit2i_cond_ctx) {
        auto* miniT2i = static_cast<MiniT2IConditioner*>(handle->minit2i_cond_ctx);
        delete miniT2i;
        handle->minit2i_cond_ctx = nullptr;
    }
    if (handle->llm_ctx) {
        auto* llm = static_cast<LLMEmbedder*>(handle->llm_ctx);
        delete llm;
        handle->llm_ctx = nullptr;
    }
    if (handle->sd3_cond_ctx) {
        auto* sd3 = static_cast<SD3CLIPEmbedder*>(handle->sd3_cond_ctx);
        delete sd3;
        handle->sd3_cond_ctx = nullptr;
    }
#endif
    handle->model_manager.reset();
    if (handle->params_backend) {
        ggml_backend_free(static_cast<ggml_backend_t>(handle->params_backend));
        handle->params_backend = nullptr;
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
