/**
 * JNI bindings for bark.cpp - Text-to-Speech synthesis
 *
 * This provides the native interface for the Bark text-to-speech model,
 * enabling high-quality voice synthesis from text input.
 */

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cmath>

#include "jni_utils.h"
#include "jni_thread_cache.h"

#if __has_include(<android/log.h>)
#include <android/log.h>
#else
#include <cstdio>
#include <cstdarg>
#define ANDROID_LOG_DEBUG 3
#define ANDROID_LOG_INFO 4
#define ANDROID_LOG_WARN 5
#define ANDROID_LOG_ERROR 6
inline int __android_log_print(int level, const char* tag, const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[%s] ", tag);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(args);
    return 0;
}
#endif

#include "bark.h"

#define LOG_TAG "BarkJNI"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Handle structure to hold bark context and JVM references
struct BarkHandle {
    bark_context* ctx = nullptr;
    JavaVM* jvm = nullptr;
    jobject progressCallbackGlobalRef = nullptr;
    jmethodID progressMethodID = nullptr;
    std::mutex mutex;
    int sampleRate = 24000; // Bark default sample rate
};

static void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    llmedge_throw_java_exception(env, className, message);
}

static BarkHandle* requireBarkHandle(JNIEnv* env, jlong handlePtr, const char* message) {
    auto* handle = reinterpret_cast<BarkHandle*>(handlePtr);
    if (!handle || !handle->ctx) {
        llmedge_throw_java_exception(env, "java/lang/IllegalStateException", message);
        return nullptr;
    }
    return handle;
}

// Progress callback wrapper — throttled to fire every 5% change per step
static void bark_progress_callback_wrapper(struct bark_context* bctx,
                                           enum bark_encoding_step step,
                                           int progress,
                                           void* user_data) {
    (void)bctx;

    auto* handle = static_cast<BarkHandle*>(user_data);
    if (!handle || !handle->progressCallbackGlobalRef || !handle->jvm || !handle->progressMethodID) {
        return;
    }

    // Throttle: only fire callback on 5% boundaries
    static thread_local int lastStep = -1;
    static thread_local int lastProgress = -1;
    jint stepInt = static_cast<jint>(step);
    if (stepInt < lastStep || (stepInt == lastStep && progress < lastProgress)) {
        lastStep = -1;
        lastProgress = -1;
    }
    if (stepInt == lastStep && progress / 5 == lastProgress / 5 && progress != 100) {
        return;
    }
    lastStep = stepInt;
    lastProgress = progress;

    JNIEnv* env = jni_thread_cache_get_env();
    if (!env) return;

    env->CallVoidMethod(handle->progressCallbackGlobalRef, handle->progressMethodID,
                        stepInt, static_cast<jint>(progress));

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeCheckBindings(JNIEnv*, jclass) {
    return JNI_TRUE;
}

JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeCreate(JNIEnv* env, jclass,
                                               jstring jModelPath,
                                               jint seed,
                                               jfloat temp,
                                               jfloat fineTemp,
                                               jint verbosity) {
    if (!jModelPath) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Model path cannot be null");
        return 0;
    }

    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    if (!modelPath) {
        throwJavaException(env, "java/lang/RuntimeException", "Failed to get model path string");
        return 0;
    }

    ALOGI("Initializing Bark with model: %s, seed=%d, temp=%.2f, fineTemp=%.2f",
          modelPath, seed, temp, fineTemp);

    bark_context_params cparams = bark_context_default_params();
    cparams.verbosity = static_cast<bark_verbosity_level>(verbosity);
    cparams.temp = temp;
    cparams.fine_temp = fineTemp;

    // Create handle first to set up callback
    auto* handle = new BarkHandle();
    env->GetJavaVM(&handle->jvm);
    jni_thread_cache_init(handle->jvm);

    // Set callback in params
    cparams.progress_callback = bark_progress_callback_wrapper;
    cparams.progress_callback_user_data = handle;

    bark_context* ctx = bark_load_model(modelPath, cparams, static_cast<uint32_t>(seed));
    env->ReleaseStringUTFChars(jModelPath, modelPath);

    if (!ctx) {
        delete handle;
        throwJavaException(env, "java/lang/RuntimeException", "Failed to initialize bark context");
        return 0;
    }

    handle->ctx = ctx;
    handle->sampleRate = cparams.sample_rate;

    ALOGI("Bark context created successfully, handle=%p, sampleRate=%d", handle, handle->sampleRate);
    return reinterpret_cast<jlong>(handle);
}

JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeDestroy(JNIEnv* env, jclass, jlong handlePtr) {
    auto* handle = reinterpret_cast<BarkHandle*>(handlePtr);
    if (!handle) return;

    // Release the mutex before deleting the handle: deleting while the
    // lock_guard still holds handle->mutex destroys a locked mutex (UB).
    {
        std::lock_guard<std::mutex> lock(handle->mutex);

        llmedge_clear_global_ref(env, handle->progressCallbackGlobalRef);

        if (handle->ctx) {
            bark_free(handle->ctx);
            handle->ctx = nullptr;
        }
    }

    delete handle;
    ALOGI("Bark context destroyed");
}

JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeSetProgressCallback(JNIEnv* env, jclass,
                                                            jlong handlePtr,
                                                            jobject callback) {
    auto* handle = requireBarkHandle(env, handlePtr, "Bark context not initialized");
    if (!handle) return;

    std::lock_guard<std::mutex> lock(handle->mutex);

    // Clear existing callback
    if (handle->progressCallbackGlobalRef) {
        llmedge_clear_global_ref(env, handle->progressCallbackGlobalRef);
        handle->progressMethodID = nullptr;
    }

    if (callback) {
        handle->progressCallbackGlobalRef =
                llmedge_new_global_ref_or_throw(
                        env,
                        callback,
                        "Unable to hold Bark progress callback reference");
        if (!handle->progressCallbackGlobalRef) {
            return;
        }
        handle->progressMethodID =
                llmedge_get_callback_method(
                        env,
                        callback,
                        "onProgress",
                        "(II)V",
                        "java/lang/NoSuchMethodError",
                        "onProgress(II)V method not found");
        if (!handle->progressMethodID) {
            llmedge_clear_global_ref(env, handle->progressCallbackGlobalRef);
            ALOGE("Failed to find onProgress(II)V method on Bark progress callback");
        }
    }
}

JNIEXPORT jfloatArray JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeGenerate(JNIEnv* env, jclass,
                                                 jlong handlePtr,
                                                 jstring jText,
                                                 jint nThreads) {
    auto* handle = requireBarkHandle(env, handlePtr, "Bark context not initialized");
    if (!handle) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(handle->mutex);

    if (!jText) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Text cannot be null");
        return nullptr;
    }

    const char* text = env->GetStringUTFChars(jText, nullptr);
    if (!text) {
        throwJavaException(env, "java/lang/RuntimeException", "Failed to get text string");
        return nullptr;
    }

    ALOGI("Generating audio for text: \"%s\", threads=%d", text, nThreads);

    // Generate audio
    bool success = bark_generate_audio(handle->ctx, text, nThreads);
    env->ReleaseStringUTFChars(jText, text);

    if (!success) {
        ALOGE("Failed to generate audio");
        throwJavaException(env, "java/lang/RuntimeException", "Failed to generate audio");
        return nullptr;
    }

    // Get audio data
    float* audioData = bark_get_audio_data(handle->ctx);
    int audioSize = bark_get_audio_data_size(handle->ctx);

    if (!audioData || audioSize <= 0) {
        ALOGE("No audio data generated");
        throwJavaException(env, "java/lang/RuntimeException", "No audio data generated");
        return nullptr;
    }

    ALOGI("Generated %d audio samples", audioSize);

    // Create Java float array and copy data
    jfloatArray result = env->NewFloatArray(audioSize);
    if (!result) {
        throwJavaException(env, "java/lang/OutOfMemoryError", "Failed to allocate audio array");
        return nullptr;
    }

    env->SetFloatArrayRegion(result, 0, audioSize, audioData);

    return result;
}

JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeGetSampleRate(JNIEnv* env, jclass, jlong handlePtr) {
    auto* handle = requireBarkHandle(env, handlePtr, "Bark context not initialized");
    if (!handle) return 24000;
    return handle->sampleRate;
}

JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeGetLoadTime(JNIEnv* env, jclass, jlong handlePtr) {
    auto* handle = requireBarkHandle(env, handlePtr, "Bark context not initialized");
    if (!handle) return 0;
    return bark_get_load_time(handle->ctx);
}

JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeGetEvalTime(JNIEnv* env, jclass, jlong handlePtr) {
    auto* handle = requireBarkHandle(env, handlePtr, "Bark context not initialized");
    if (!handle) return 0;
    return bark_get_eval_time(handle->ctx);
}

JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_speech_tts_BarkTTS_nativeResetStatistics(JNIEnv* env, jclass, jlong handlePtr) {
    auto* handle = requireBarkHandle(env, handlePtr, "Bark context not initialized");
    if (!handle) return;
    bark_reset_statistics(handle->ctx);
}

} // extern "C"
