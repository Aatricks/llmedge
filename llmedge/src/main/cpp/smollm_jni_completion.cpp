#include "smollm_jni_shared.h"

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_addChatMessage(JNIEnv* env, jobject thiz, jlong modelPtr, jstring message,
                                                  jstring role) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    ScopedUtfChars messageCstr(env, message);
    ScopedUtfChars roleCstr(env, role);
    if (!messageCstr.ok() || !roleCstr.ok()) {
        return;
    }
    try {
        llmInference->addChatMessage(messageCstr.get(), roleCstr.get());
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_startCompletion(JNIEnv* env, jobject thiz, jlong modelPtr, jstring prompt) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    ScopedUtfChars promptCstr(env, prompt);
    if (!promptCstr.ok()) {
        return;
    }
    try {
        llmInference->startCompletion(promptCstr.get());
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_setReasoningOptions(JNIEnv* env, jobject thiz, jlong modelPtr, jboolean disableThinking,
                                                    jint reasoningBudget) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    const bool disable = disableThinking == JNI_TRUE;
    llmInference->setReasoningOptions(disable, reasoningBudget);
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_completionLoop(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    try {
        std::string response = llmInference->completionLoop();
        return env->NewStringUTF(response.c_str());
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_completionLoopBatch(JNIEnv* env, jobject thiz, jlong modelPtr, jint maxTokens) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    try {
        std::string response = llmInference->completionLoopBatch(maxTokens);
        return env->NewStringUTF(response.c_str());
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_completionLoopBatchBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jint maxTokens) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    try {
        std::string response = llmInference->completionLoopBatch(maxTokens);
        jbyteArray result = env->NewByteArray(static_cast<jsize>(response.size()));
        if (result && !response.empty()) {
            env->SetByteArrayRegion(result, 0, static_cast<jsize>(response.size()),
                                    reinterpret_cast<const jbyte*>(response.c_str()));
        }
        return result;
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_setThreadAffinity(JNIEnv* env, jobject thiz, jlong modelPtr, jlong coreMask) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (llmInference) {
        llmInference->setThreadAffinity(static_cast<uint64_t>(coreMask));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_stopCompletion(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    llmInference->stopCompletion();
}
