#include "smollm_jni_shared.h"

#include <vector>

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) return nullptr;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return nullptr;
    size_t size = llama_state_get_size(ctx);
    if (size == 0) return nullptr;
    std::vector<uint8_t> buf(size);
    size_t written = llama_state_get_data(ctx, buf.data(), buf.size());
    if (written == 0) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(written));
    if (!arr) return nullptr;
    env->SetByteArrayRegion(arr, 0, static_cast<jsize>(written), reinterpret_cast<jbyte*>(buf.data()));
    return arr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeSetStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jbyteArray state) {
    if (!modelPtr || !state) return JNI_FALSE;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return JNI_FALSE;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return JNI_FALSE;
    jsize len = env->GetArrayLength(state);
    if (len <= 0) return JNI_FALSE;
    std::vector<uint8_t> buf(static_cast<size_t>(len));
    env->GetByteArrayRegion(state, 0, len, reinterpret_cast<jbyte*>(buf.data()));
    size_t written = llama_state_set_data(ctx, buf.data(), buf.size());
    return (written == static_cast<size_t>(len)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetSequenceStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jint seqId) {
    if (!modelPtr) return nullptr;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return nullptr;
    size_t size = llmedge_state_seq_get_size(ctx, static_cast<llama_seq_id>(seqId));
    if (size == 0) return nullptr;
    std::vector<uint8_t> buf(size);
    size_t written = llmedge_state_seq_get_data(ctx, buf.data(), buf.size(), static_cast<llama_seq_id>(seqId));
    if (written == 0) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(written));
    if (!arr) return nullptr;
    env->SetByteArrayRegion(arr, 0, static_cast<jsize>(written), reinterpret_cast<jbyte*>(buf.data()));
    return arr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeSetSequenceStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jint seqId, jbyteArray state) {
    if (!modelPtr || !state) return JNI_FALSE;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return JNI_FALSE;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return JNI_FALSE;
    jsize len = env->GetArrayLength(state);
    if (len <= 0) return JNI_FALSE;
    std::vector<uint8_t> buf(static_cast<size_t>(len));
    env->GetByteArrayRegion(state, 0, len, reinterpret_cast<jbyte*>(buf.data()));
    size_t written = llmedge_state_seq_set_data(ctx, buf.data(), buf.size(), static_cast<llama_seq_id>(seqId));
    return (written == static_cast<size_t>(len)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeClearKvCache(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) return;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return;
    llmedge_kv_cache_clear(ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeClearMessages(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) return;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    llmInference->clearMessages();
}
