#include "gguf_reader_internal.h"

#include <jni.h>
#include <string>

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

// CPU feature probe for native-library variant selection. Lives in the GGUF
// reader library because it is always compiled for the baseline ISA and is
// therefore safe to load on any device before feature detection has run.
// Returns [AT_HWCAP, AT_HWCAP2] on aarch64 Linux/Android, [0, 0] elsewhere.
extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_core_NativeCpuFeatures_nativeGetHwcaps(JNIEnv* env, jclass) {
    jlong caps[2] = {0, 0};
#if defined(__aarch64__) && defined(__linux__)
    caps[0] = static_cast<jlong>(getauxval(AT_HWCAP));
    caps[1] = static_cast<jlong>(getauxval(AT_HWCAP2));
#endif
    jlongArray result = env->NewLongArray(2);
    if (result) {
        env->SetLongArrayRegion(result, 0, 2, caps);
    }
    return result;
}

namespace {
// GGUF metadata is untrusted file content and may contain 4-byte UTF-8
// (invalid Modified UTF-8), so it must not go through NewStringUTF.
// Return raw UTF-8 bytes and decode on the Kotlin side instead.
jbyteArray to_utf8_bytes(JNIEnv* env, const std::string& value) {
    jbyteArray result = env->NewByteArray(static_cast<jsize>(value.size()));
    if (result && !value.empty()) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(value.size()),
                                reinterpret_cast<const jbyte*>(value.data()));
    }
    return result;
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getGGUFContextNativeHandle(JNIEnv* env, jobject thiz, jstring modelPath) {
    jboolean         isCopy        = true;
    const char*      modelPathCStr = env->GetStringUTFChars(modelPath, &isCopy);
    if (modelPathCStr == nullptr) {
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        if (exClass != nullptr) {
            env->ThrowNew(exClass, "Failed to get model path characters");
        }
        return 0;
    }
    gguf_context*    ggufContext   = llmedge_gguf_open_file(modelPathCStr);
    env->ReleaseStringUTFChars(modelPath, modelPathCStr);
    return reinterpret_cast<jlong>(ggufContext);
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getContextSize(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    return llmedge_gguf_get_context_size(ggufContext);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getChatTemplateBytes(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    std::string   chatTemplate = llmedge_gguf_get_chat_template(ggufContext);
    return to_utf8_bytes(env, chatTemplate);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getArchitectureBytes(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext       = reinterpret_cast<gguf_context*>(nativeHandle);
    std::string architecture = llmedge_gguf_get_architecture(ggufContext);
    return to_utf8_bytes(env, architecture);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getParameterCountBytes(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    const std::string paramCount = llmedge_gguf_get_parameter_count(ggufContext);
    return to_utf8_bytes(env, paramCount);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getModelNameBytes(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    const std::string modelName = llmedge_gguf_get_model_name(ggufContext);
    return to_utf8_bytes(env, modelName);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_runtime_GGUFReader_00024DefaultNativeBridge_getTensorNamePrefixesBytes(JNIEnv* env, jobject thiz, jlong nativeHandle) {
    gguf_context* ggufContext = reinterpret_cast<gguf_context*>(nativeHandle);
    const std::string prefixes = llmedge_gguf_get_tensor_name_prefixes(ggufContext);
    return to_utf8_bytes(env, prefixes);
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
