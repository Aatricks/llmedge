#include "smollm_jni_shared.h"

extern "C" JNIEXPORT jfloat JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getResponseGenerationSpeed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0.0f;
    }
    return llmInference->getResponseGenerationTime();
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getResponseGeneratedTokenCount(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return llmInference->getResponseTokenCount();
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getResponseGenerationDurationMicros(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return llmInference->getResponseGenerationTimeMicros();
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetLastGenerationMetrics(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }

    const int64_t elapsedMicros = llmInference->getResponseGenerationTimeMicros();
    const long tokenCount = llmInference->getResponseTokenCount();
    const float tokensPerSecond = llmInference->getResponseTokensPerSecond();

    uint32_t speedBits = 0;
    static_assert(sizeof(speedBits) == sizeof(tokensPerSecond), "Unexpected float size");
    std::memcpy(&speedBits, &tokensPerSecond, sizeof(tokensPerSecond));

    jlong values[3] = {
        static_cast<jlong>(elapsedMicros),
        static_cast<jlong>(tokenCount),
        static_cast<jlong>(speedBits),
    };
    jlongArray result = env->NewLongArray(3);
    if (!result) {
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 3, values);
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getContextSizeUsed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return llmInference->getContextSizeUsed();
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeConfigureThreading(JNIEnv* env, jobject thiz, jlong modelPtr,
                                                         jint generationThreads, jint promptThreads) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    llmInference->configureThreading(generationThreads, promptThreads);
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetEstimatedMemoryBytes(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return static_cast<jlong>(llmInference->getEstimatedMemoryBytes());
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetEstimatedStateMemoryBytes(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return static_cast<jlong>(llmInference->getStateMemoryBytes());
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_close(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) {
        return;
    }
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    delete llmInference;
}
