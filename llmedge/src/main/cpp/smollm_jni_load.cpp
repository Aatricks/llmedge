#include "smollm_jni_shared.h"

#include <memory>
#include <cstring>

#include "ggml-backend.h"

namespace {

bool has_registered_backend_prefix(const char* prefix) {
    const size_t prefix_len = std::strlen(prefix);
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        const char* reg_name = ggml_backend_reg_get_name(i);
        if (reg_name && std::strncmp(reg_name, prefix, prefix_len) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_loadModel(JNIEnv* env, jobject thiz, jstring modelPath, jfloat minP,
                                             jfloat temperature, jboolean storeChats, jlong contextSize,
                                             jstring chatTemplate, jint nThreads, jboolean useMmap, jboolean useMlock,
                                             jint backendId, jboolean useFlashAttn, jint kvCacheTypeK, jint kvCacheTypeV,
                                             jint nGpuLayers, jint nUbatch) {
    ScopedUtfChars modelPathCstr(env, modelPath);
    ScopedUtfChars chatTemplateCstr(env, chatTemplate);
    if (!modelPathCstr.ok() || !chatTemplateCstr.ok()) {
        return 0;
    }

    auto llmInference = std::make_unique<LLMInference>();
    try {
        llmInference->loadModel(modelPathCstr.get(), minP, temperature, storeChats, contextSize, chatTemplateCstr.get(), nThreads,
                                useMmap, useMlock, backendId, useFlashAttn, kvCacheTypeK, kvCacheTypeV, nGpuLayers, nUbatch);
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return 0;
    }
    return reinterpret_cast<jlong>(llmInference.release());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeHasVulkanBackendSupport(JNIEnv* env, jobject thiz) {
#ifdef GGML_USE_VULKAN
    return JNI_TRUE;
#else
    (void) env;
    (void) thiz;
    return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeIsOpenClAvailable(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
#ifdef GGML_USE_OPENCL
    return has_registered_backend_prefix("OpenCL") ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeIsVulkanAvailable(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
#ifdef GGML_USE_VULKAN
    return has_registered_backend_prefix("Vulkan") ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_FALSE;
#endif
}
