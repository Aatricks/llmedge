#pragma once

#include <jni.h>
#include <atomic>

struct sd_ctx_t;

struct SdHandle {
    sd_ctx_t* ctx = nullptr;
    void* t5_ctx = nullptr; // Pointer to T5CLIPEmbedder for T5-only mode
    float flowShift = 0.0f;
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
#define SD_JNI_INTERNAL static
inline void sd_jni_notify_frame_buffer_freed(const void*) {}
inline void sd_jni_notify_frame_array_freed(const void*) {}
#endif

SD_JNI_INTERNAL void clearProgressCallback(JNIEnv* env, SdHandle* handle);
SD_JNI_INTERNAL void sd_video_progress_wrapper(int step, int steps, float time, void* data);
SD_JNI_INTERNAL void throwJavaException(JNIEnv* env, const char* className, const char* message);
