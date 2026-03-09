/**
 * JNI bindings for whisper.cpp - Speech-to-Text transcription
 *
 * This provides the native interface for the Whisper speech recognition model,
 * enabling real-time transcription, translation, and subtitle generation.
 */

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <cstring>

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

#include "whisper.h"

#define LOG_TAG "WhisperJNI"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Handle structure to hold whisper context and JVM references
struct WhisperHandle {
    whisper_context* ctx = nullptr;
    JavaVM* jvm = nullptr;
    jobject progressCallbackGlobalRef = nullptr;
    jmethodID progressMethodID = nullptr;
    jobject segmentCallbackGlobalRef = nullptr;
    jmethodID segmentMethodID = nullptr;
    std::mutex mutex;
};

static void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    if (!env) return;
    jclass exClass = env->FindClass(className);
    if (!exClass) return;
    env->ThrowNew(exClass, message);
}

static WhisperHandle* requireWhisperHandle(JNIEnv* env, jlong handlePtr, const char* message) {
    auto* handle = reinterpret_cast<WhisperHandle*>(handlePtr);
    if (!handle || !handle->ctx) {
        throwJavaException(env, "java/lang/IllegalStateException", message);
        return nullptr;
    }
    return handle;
}

// Progress callback wrapper — throttled to fire every 5% change
static void whisper_progress_callback_wrapper(struct whisper_context* ctx,
                                               struct whisper_state* state,
                                               int progress,
                                               void* user_data) {
    (void)ctx;
    (void)state;

    auto* handle = static_cast<WhisperHandle*>(user_data);
    if (!handle || !handle->progressCallbackGlobalRef || !handle->jvm || !handle->progressMethodID) {
        return;
    }

    // Throttle: only fire callback on 5% boundaries
    static thread_local int lastReportedProgress = -1;
    if (progress / 5 == lastReportedProgress / 5 && progress != 100) {
        return;
    }
    lastReportedProgress = progress;

    JNIEnv* env = jni_thread_cache_get_env();
    if (!env) return;

    env->CallVoidMethod(handle->progressCallbackGlobalRef, handle->progressMethodID,
                        static_cast<jint>(progress));

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}

// New segment callback wrapper
static void whisper_new_segment_callback_wrapper(struct whisper_context* ctx,
                                                  struct whisper_state* state,
                                                  int n_new,
                                                  void* user_data) {
    auto* handle = static_cast<WhisperHandle*>(user_data);
    if (!handle || !handle->segmentCallbackGlobalRef || !handle->jvm || !handle->segmentMethodID) {
        return;
    }

    JNIEnv* env = jni_thread_cache_get_env();
    if (!env) return;

    // Get segment count
    int n_segments = whisper_full_n_segments_from_state(state);

    // Call callback for each new segment
    for (int i = n_segments - n_new; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text_from_state(state, i);
        int64_t t0 = whisper_full_get_segment_t0_from_state(state, i);
        int64_t t1 = whisper_full_get_segment_t1_from_state(state, i);

        jstring jText = env->NewStringUTF(text ? text : "");
        env->CallVoidMethod(handle->segmentCallbackGlobalRef, handle->segmentMethodID,
                            static_cast<jint>(i),
                            static_cast<jlong>(t0),
                            static_cast<jlong>(t1),
                            jText);
        env->DeleteLocalRef(jText);

        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            break;
        }
    }
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeCheckBindings(JNIEnv*, jclass) {
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeGetVersion(JNIEnv* env, jclass) {
    const char* version = whisper_version();
    return env->NewStringUTF(version ? version : "unknown");
}

JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeGetSystemInfo(JNIEnv* env, jclass) {
    const char* info = whisper_print_system_info();
    return env->NewStringUTF(info ? info : "");
}

JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeCreate(JNIEnv* env, jclass,
                                               jstring jModelPath,
                                               jboolean useGpu,
                                               jboolean flashAttn,
                                               jint gpuDevice) {
    if (!jModelPath) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Model path cannot be null");
        return 0;
    }

    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    if (!modelPath) {
        throwJavaException(env, "java/lang/RuntimeException", "Failed to get model path string");
        return 0;
    }

    ALOGI("Initializing Whisper with model: %s, useGpu=%d, flashAttn=%d, gpuDevice=%d",
          modelPath, useGpu, flashAttn, gpuDevice);

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = useGpu;
    cparams.flash_attn = flashAttn;
    cparams.gpu_device = gpuDevice;

    whisper_context* ctx = whisper_init_from_file_with_params(modelPath, cparams);
    env->ReleaseStringUTFChars(jModelPath, modelPath);

    if (!ctx) {
        throwJavaException(env, "java/lang/RuntimeException", "Failed to initialize whisper context");
        return 0;
    }

    auto* handle = new WhisperHandle();
    handle->ctx = ctx;
    env->GetJavaVM(&handle->jvm);
    jni_thread_cache_init(handle->jvm);

    ALOGI("Whisper context created successfully, handle=%p", handle);
    return reinterpret_cast<jlong>(handle);
}

JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeDestroy(JNIEnv* env, jclass, jlong handlePtr) {
    auto* handle = reinterpret_cast<WhisperHandle*>(handlePtr);
    if (!handle) return;

    std::lock_guard<std::mutex> lock(handle->mutex);

    if (handle->progressCallbackGlobalRef && env) {
        env->DeleteGlobalRef(handle->progressCallbackGlobalRef);
    }
    if (handle->segmentCallbackGlobalRef && env) {
        env->DeleteGlobalRef(handle->segmentCallbackGlobalRef);
    }

    if (handle->ctx) {
        whisper_free(handle->ctx);
    }

    delete handle;
    ALOGI("Whisper context destroyed");
}

JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeGetMaxLanguageId(JNIEnv*, jclass) {
    return whisper_lang_max_id();
}

JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeGetLanguageId(JNIEnv* env, jclass, jstring jLang) {
    if (!jLang) return -1;
    const char* lang = env->GetStringUTFChars(jLang, nullptr);
    if (!lang) return -1;
    int id = whisper_lang_id(lang);
    env->ReleaseStringUTFChars(jLang, lang);
    return id;
}

JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeGetLanguageString(JNIEnv* env, jclass, jint langId) {
    const char* lang = whisper_lang_str(langId);
    return env->NewStringUTF(lang ? lang : "");
}

JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeIsMultilingual(JNIEnv*, jclass, jlong handlePtr) {
    auto* handle = reinterpret_cast<WhisperHandle*>(handlePtr);
    if (!handle || !handle->ctx) return JNI_FALSE;
    return whisper_is_multilingual(handle->ctx) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeGetModelType(JNIEnv* env, jclass, jlong handlePtr) {
    auto* handle = requireWhisperHandle(env, handlePtr, "Whisper context not initialized");
    if (!handle) {
        return nullptr;
    }
    const char* type = whisper_model_type_readable(handle->ctx);
    return env->NewStringUTF(type ? type : "unknown");
}

JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeSetProgressCallback(JNIEnv* env, jclass,
                                                            jlong handlePtr,
                                                            jobject callback) {
    auto* handle = requireWhisperHandle(env, handlePtr, "Whisper context not initialized");
    if (!handle) return;

    std::lock_guard<std::mutex> lock(handle->mutex);

    // Clear existing callback
    if (handle->progressCallbackGlobalRef) {
        env->DeleteGlobalRef(handle->progressCallbackGlobalRef);
        handle->progressCallbackGlobalRef = nullptr;
        handle->progressMethodID = nullptr;
    }

    if (callback) {
        handle->progressCallbackGlobalRef = env->NewGlobalRef(callback);
        jclass callbackClass = env->GetObjectClass(callback);
        handle->progressMethodID = env->GetMethodID(callbackClass, "onProgress", "(I)V");
        env->DeleteLocalRef(callbackClass);
    }
}

JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeSetSegmentCallback(JNIEnv* env, jclass,
                                                           jlong handlePtr,
                                                           jobject callback) {
    auto* handle = requireWhisperHandle(env, handlePtr, "Whisper context not initialized");
    if (!handle) return;

    std::lock_guard<std::mutex> lock(handle->mutex);

    // Clear existing callback
    if (handle->segmentCallbackGlobalRef) {
        env->DeleteGlobalRef(handle->segmentCallbackGlobalRef);
        handle->segmentCallbackGlobalRef = nullptr;
        handle->segmentMethodID = nullptr;
    }

    if (callback) {
        handle->segmentCallbackGlobalRef = env->NewGlobalRef(callback);
        jclass callbackClass = env->GetObjectClass(callback);
        handle->segmentMethodID = env->GetMethodID(callbackClass, "onNewSegment", "(IJJLjava/lang/String;)V");
        env->DeleteLocalRef(callbackClass);
    }
}

JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeTranscribe(JNIEnv* env, jclass,
                                                   jlong handlePtr,
                                                   jfloatArray jSamples,
                                                   jint nThreads,
                                                   jboolean translate,
                                                   jstring jLanguage,
                                                   jboolean detectLanguage,
                                                   jboolean tokenTimestamps,
                                                   jint maxLen,
                                                   jboolean splitOnWord,
                                                   jfloat temperature,
                                                   jint beamSize,
                                                   jboolean suppressBlank,
                                                   jboolean printProgress) {
    auto* handle = requireWhisperHandle(env, handlePtr, "Whisper context not initialized");
    if (!handle) return nullptr;

    if (!jSamples) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Audio samples cannot be null");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(handle->mutex);

    jint n_samples = env->GetArrayLength(jSamples);
    jfloat* samples = env->GetFloatArrayElements(jSamples, nullptr);
    if (!samples) {
        throwJavaException(env, "java/lang/RuntimeException", "Failed to get audio samples");
        return nullptr;
    }

    const char* language = jLanguage ? env->GetStringUTFChars(jLanguage, nullptr) : nullptr;

    // Set up whisper parameters
    whisper_full_params wparams = whisper_full_default_params(
        beamSize > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    wparams.n_threads = nThreads > 0 ? nThreads : 4;
    wparams.translate = translate;
    wparams.language = language;
    wparams.detect_language = detectLanguage;
    wparams.token_timestamps = tokenTimestamps;
    wparams.max_len = maxLen;
    wparams.split_on_word = splitOnWord;
    wparams.temperature = temperature;
    wparams.suppress_blank = suppressBlank;
    wparams.print_progress = printProgress;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;

    if (beamSize > 1) {
        wparams.beam_search.beam_size = beamSize;
    }

    // Set progress callback if registered
    if (handle->progressCallbackGlobalRef) {
        wparams.progress_callback = whisper_progress_callback_wrapper;
        wparams.progress_callback_user_data = handle;
    }

    // Set segment callback if registered
    if (handle->segmentCallbackGlobalRef) {
        wparams.new_segment_callback = whisper_new_segment_callback_wrapper;
        wparams.new_segment_callback_user_data = handle;
    }

    ALOGI("Starting transcription: samples=%d, threads=%d, translate=%d, language=%s",
          n_samples, wparams.n_threads, translate, language ? language : "auto");

    int result = whisper_full(handle->ctx, wparams, samples, n_samples);

    env->ReleaseFloatArrayElements(jSamples, samples, JNI_ABORT);
    if (language) {
        env->ReleaseStringUTFChars(jLanguage, language);
    }

    if (result != 0) {
        throwJavaException(env, "java/lang/RuntimeException", "Transcription failed");
        return nullptr;
    }

    // Collect segments
    int n_segments = whisper_full_n_segments(handle->ctx);
    ALOGI("Transcription complete: %d segments", n_segments);

    // Create TranscriptionSegment array
    jclass segmentClass = env->FindClass("io/aatricks/llmedge/Whisper$TranscriptionSegment");
    if (!segmentClass) {
        throwJavaException(env, "java/lang/RuntimeException", "TranscriptionSegment class not found");
        return nullptr;
    }

    jmethodID segmentCtor = env->GetMethodID(segmentClass, "<init>", "(IJJLjava/lang/String;)V");
    if (!segmentCtor) {
        throwJavaException(env, "java/lang/RuntimeException", "TranscriptionSegment constructor not found");
        return nullptr;
    }

    jobjectArray segmentArray = env->NewObjectArray(n_segments, segmentClass, nullptr);
    if (!segmentArray) {
        return nullptr;
    }

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(handle->ctx, i);
        int64_t t0 = whisper_full_get_segment_t0(handle->ctx, i);
        int64_t t1 = whisper_full_get_segment_t1(handle->ctx, i);

        jstring jText = env->NewStringUTF(text ? text : "");
        jobject segment = env->NewObject(segmentClass, segmentCtor,
                                          static_cast<jint>(i),
                                          static_cast<jlong>(t0),
                                          static_cast<jlong>(t1),
                                          jText);
        env->SetObjectArrayElement(segmentArray, i, segment);
        env->DeleteLocalRef(jText);
        env->DeleteLocalRef(segment);
    }

    return segmentArray;
}

JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeDetectLanguage(JNIEnv* env, jclass,
                                                       jlong handlePtr,
                                                       jfloatArray jSamples,
                                                       jint nThreads,
                                                       jint offsetMs) {
    auto* handle = requireWhisperHandle(env, handlePtr, "Whisper context not initialized");
    if (!handle) return -1;

    if (!jSamples) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Audio samples cannot be null");
        return -1;
    }

    std::lock_guard<std::mutex> lock(handle->mutex);

    jint n_samples = env->GetArrayLength(jSamples);
    jfloat* samples = env->GetFloatArrayElements(jSamples, nullptr);
    if (!samples) {
        return -1;
    }

    // First, we need to compute the mel spectrogram
    int result = whisper_pcm_to_mel(handle->ctx, samples, n_samples, nThreads > 0 ? nThreads : 4);
    env->ReleaseFloatArrayElements(jSamples, samples, JNI_ABORT);

    if (result != 0) {
        ALOGE("Failed to compute mel spectrogram for language detection");
        return -1;
    }

    // Detect language
    int langId = whisper_lang_auto_detect(handle->ctx, offsetMs, nThreads > 0 ? nThreads : 4, nullptr);

    ALOGI("Detected language ID: %d (%s)", langId, langId >= 0 ? whisper_lang_str(langId) : "unknown");
    return langId;
}

JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeGetFullText(JNIEnv* env, jclass, jlong handlePtr) {
    auto* handle = requireWhisperHandle(env, handlePtr, "Whisper context not initialized");
    if (!handle) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(handle->mutex);

    int n_segments = whisper_full_n_segments(handle->ctx);
    std::string fullText;

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(handle->ctx, i);
        if (text) {
            fullText += text;
        }
    }

    return env->NewStringUTF(fullText.c_str());
}

JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativeResetTimings(JNIEnv*, jclass, jlong handlePtr) {
    auto* handle = reinterpret_cast<WhisperHandle*>(handlePtr);
    if (!handle || !handle->ctx) return;
    whisper_reset_timings(handle->ctx);
}

JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_speech_stt_Whisper_nativePrintTimings(JNIEnv*, jclass, jlong handlePtr) {
    auto* handle = reinterpret_cast<WhisperHandle*>(handlePtr);
    if (!handle || !handle->ctx) return;
    whisper_print_timings(handle->ctx);
}

} // extern "C"
