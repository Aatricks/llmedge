#include "whisper_jni_common.h"

#include "jni_thread_cache.h"
#include "jni_utils.h"

void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    llmedge_throw_java_exception(env, className, message);
}

WhisperHandle* requireWhisperHandle(JNIEnv* env, jlong handlePtr, const char* message) {
    auto* handle = reinterpret_cast<WhisperHandle*>(handlePtr);
    if (!handle || !handle->ctx) {
        llmedge_throw_java_exception(env, "java/lang/IllegalStateException", message);
        return nullptr;
    }
    return handle;
}

void whisper_progress_callback_wrapper(
    struct whisper_context* ctx,
    struct whisper_state* state,
    int progress,
    void* user_data
) {
    (void)ctx;
    (void)state;

    auto* handle = static_cast<WhisperHandle*>(user_data);
    if (!handle || !handle->progressCallbackGlobalRef || !handle->jvm || !handle->progressMethodID) {
        return;
    }

    static thread_local int lastReportedProgress = -1;
    if (progress / 5 == lastReportedProgress / 5 && progress != 100) {
        return;
    }
    lastReportedProgress = progress;

    JNIEnv* env = jni_thread_cache_get_env();
    if (!env) {
        return;
    }

    env->CallVoidMethod(handle->progressCallbackGlobalRef, handle->progressMethodID, static_cast<jint>(progress));

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}

void whisper_new_segment_callback_wrapper(
    struct whisper_context* ctx,
    struct whisper_state* state,
    int n_new,
    void* user_data
) {
    (void)ctx;

    auto* handle = static_cast<WhisperHandle*>(user_data);
    if (!handle || !handle->segmentCallbackGlobalRef || !handle->jvm || !handle->segmentMethodID) {
        return;
    }

    JNIEnv* env = jni_thread_cache_get_env();
    if (!env) {
        return;
    }

    int n_segments = whisper_full_n_segments_from_state(state);
    int start = n_segments - n_new;

    if (n_new > 16) {
        env->EnsureLocalCapacity(n_new + 4);
    }

    for (int i = start; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text_from_state(state, i);
        int64_t t0 = whisper_full_get_segment_t0_from_state(state, i);
        int64_t t1 = whisper_full_get_segment_t1_from_state(state, i);

        jstring jText = env->NewStringUTF(text ? text : "");
        env->CallVoidMethod(
            handle->segmentCallbackGlobalRef,
            handle->segmentMethodID,
            static_cast<jint>(i),
            static_cast<jlong>(t0),
            static_cast<jlong>(t1),
            jText
        );
        env->DeleteLocalRef(jText);

        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            break;
        }
    }
}
