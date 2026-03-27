#include "gguf_reader_internal.h"

#include <jni.h>
#include <string>

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getGGUFContextNativeHandle(JNIEnv* env, jobject thiz, jstring modelPath) {
    jboolean         isCopy        = true;
    const char*      modelPathCStr = env->GetStringUTFChars(modelPath, &isCopy);
    gguf_context*    ggufContext   = llmedge_gguf_open_file(modelPathCStr);
    env->ReleaseStringUTFChars(modelPath, modelPathCStr);
    return reinterpret_cast<jlong>(ggufContext);
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getContextSize(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    return llmedge_gguf_get_context_size(ggufContext);
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getChatTemplate(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    std::string   chatTemplate = llmedge_gguf_get_chat_template(ggufContext);
    return env->NewStringUTF(chatTemplate.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getArchitecture(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    std::string architecture = llmedge_gguf_get_architecture(ggufContext);
    return env->NewStringUTF(architecture.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getParameterCount(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    const std::string paramCount = llmedge_gguf_get_parameter_count(ggufContext);
    return env->NewStringUTF(paramCount.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getModelName(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    const std::string modelName = llmedge_gguf_get_model_name(ggufContext);
    return env->NewStringUTF(modelName.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_releaseGGUFContext(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    auto* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    if (ggufContext != nullptr) {
        gguf_free(ggufContext);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getFileType(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    return static_cast<jint>(llmedge_gguf_get_file_type(ggufContext));
}

extern "C" JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getDominantTensorType(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    return static_cast<jint>(llmedge_gguf_get_dominant_tensor_type(ggufContext));
}
