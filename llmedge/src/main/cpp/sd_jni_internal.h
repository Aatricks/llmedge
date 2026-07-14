#pragma once

#include <jni.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "model_manager.h"

struct sd_ctx_t;

struct SdHandle {
    sd_ctx_t* ctx = nullptr;
    void* t5_ctx = nullptr; // Pointer to T5CLIPEmbedder for T5-only mode
    void* minit2i_cond_ctx = nullptr; // Pointer to MiniT2IConditioner for MiniT2I sequential conditioning
    void* llm_ctx = nullptr; // Pointer to LLMEmbedder for Qwen3-only mode (FLUX.2 sequential)
    void* sd3_cond_ctx = nullptr; // Pointer to SD3CLIPEmbedder for SD3 split conditioning
    void* backend = nullptr; // Pointer to ggml_backend_t for encoder-only handles
    void* params_backend = nullptr; // Optional separate CPU params backend for encoder-only handles
    std::shared_ptr<ModelManager> model_manager;
    float flowShift = 0.0f;
    std::string loraModelDir;
    int last_width = 0;
    int last_height = 0;
    JavaVM* jvm = nullptr;
    jobject progressCallbackGlobalRef = nullptr;
    jmethodID progressMethodID = nullptr;
    std::atomic<bool> cancellationRequested{false};
    int totalFrames = 0;
    int stepsPerFrame = 0;
    int totalSteps = 0;
    int currentFrame = 0;
};

#if defined(SD_JNI_TESTING)
#define SD_JNI_INTERNAL
void sd_jni_notify_frame_buffer_freed(const void* ptr);
void sd_jni_notify_frame_array_freed(const void* ptr);
#else
#define SD_JNI_INTERNAL
inline void sd_jni_notify_frame_buffer_freed(const void*) {}
inline void sd_jni_notify_frame_array_freed(const void*) {}
#endif

SD_JNI_INTERNAL void clearProgressCallback(JNIEnv* env, SdHandle* handle);
SD_JNI_INTERNAL void sd_video_progress_wrapper(int step, int steps, float time, void* data);
SD_JNI_INTERNAL void throwJavaException(JNIEnv* env, const char* className, const char* message);
