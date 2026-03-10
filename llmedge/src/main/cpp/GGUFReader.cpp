#include "gguf.h"
#include <jni.h>
#include <string>

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getGGUFContextNativeHandle(JNIEnv* env, jobject thiz, jstring modelPath) {
    jboolean         isCopy        = true;
    const char*      modelPathCStr = env->GetStringUTFChars(modelPath, &isCopy);
    gguf_init_params initParams    = { .no_alloc = true, .ctx = nullptr };
    gguf_context*    ggufContext   = gguf_init_from_file(modelPathCStr, initParams);
    env->ReleaseStringUTFChars(modelPath, modelPathCStr);
    return reinterpret_cast<jlong>(ggufContext);
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getContextSize(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    int64_t       architectureKeyId = gguf_find_key(ggufContext, "general.architecture");
    if (architectureKeyId == -1)
        return -1;
    std::string architecture       = gguf_get_val_str(ggufContext, architectureKeyId);
    std::string contextLengthKey   = architecture + ".context_length";
    int64_t     contextLengthKeyId = gguf_find_key(ggufContext, contextLengthKey.c_str());
    if (contextLengthKeyId == -1)
        return -1;
    uint32_t contextLength = gguf_get_val_u32(ggufContext, contextLengthKeyId);
    return contextLength;
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getChatTemplate(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    int64_t       chatTemplateKeyId = gguf_find_key(ggufContext, "tokenizer.chat_template");
    std::string   chatTemplate;
    if (chatTemplateKeyId == -1) {
        chatTemplate = "";
    } else {
        chatTemplate = gguf_get_val_str(ggufContext, chatTemplateKeyId);
    }
    return env->NewStringUTF(chatTemplate.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getArchitecture(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    int64_t       architectureKeyId = gguf_find_key(ggufContext, "general.architecture");
    if (architectureKeyId == -1) {
        return env->NewStringUTF("");
    }
    std::string architecture = gguf_get_val_str(ggufContext, architectureKeyId);
    return env->NewStringUTF(architecture.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getParameterCount(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    int64_t       paramCountKeyId = gguf_find_key(ggufContext, "llama.parameter_count");
    if (paramCountKeyId == -1) {
        return env->NewStringUTF("");
    }
    uint64_t paramCount = gguf_get_val_u64(ggufContext, paramCountKeyId);
    return env->NewStringUTF(std::to_string(paramCount).c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getModelName(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    int64_t       modelNameKeyId = gguf_find_key(ggufContext, "general.name");
    if (modelNameKeyId == -1) {
        return env->NewStringUTF("");
    }
    std::string modelName = gguf_get_val_str(ggufContext, modelNameKeyId);
    return env->NewStringUTF(modelName.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_releaseGGUFContext(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    auto* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    if (ggufContext != nullptr) {
        gguf_free(ggufContext);
    }
}