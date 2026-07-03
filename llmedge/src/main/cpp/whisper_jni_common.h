#pragma once

#include <atomic>
#include <jni.h>
#include <mutex>

#include "whisper.h"

struct WhisperHandle {
    whisper_context* ctx = nullptr;
    JavaVM* jvm = nullptr;
    jobject progressCallbackGlobalRef = nullptr;
    jmethodID progressMethodID = nullptr;
    jobject segmentCallbackGlobalRef = nullptr;
    jmethodID segmentMethodID = nullptr;
    // Recursive: whisper_full invokes the segment/progress callbacks inline on the
    // transcribing thread, and a Kotlin callback calling back into any Whisper
    // method would otherwise self-deadlock on a plain mutex.
    std::recursive_mutex mutex;
    std::atomic<bool> abortRequested{false};
};

void throwJavaException(JNIEnv* env, const char* className, const char* message);
WhisperHandle* requireWhisperHandle(JNIEnv* env, jlong handlePtr, const char* message);
void whisper_progress_callback_wrapper(struct whisper_context* ctx, struct whisper_state* state, int progress, void* user_data);
void whisper_new_segment_callback_wrapper(struct whisper_context* ctx, struct whisper_state* state, int n_new, void* user_data);
