#include "sdcpp_jni_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "jni_utils.h"
#include "jni_thread_cache.h"

void free_sd_generated_frames(sd_image_t* frames, int numFrames) {
    if (!frames) {
        return;
    }
    for (int i = 0; i < numFrames; ++i) {
        if (frames[i].data) {
            sd_jni_notify_frame_buffer_freed(frames[i].data);
            free(frames[i].data);
            frames[i].data = nullptr;
        }
    }
    sd_jni_notify_frame_array_freed(frames);
    free(frames);
}

jintArray rgb_to_argb_int_array(JNIEnv* env, const uint8_t* rgb, int width, int height, int channel) {
    if (channel < 3) {
        return nullptr;
    }
    const int pixelCount = width * height;
    jintArray result = env->NewIntArray(pixelCount);
    if (!result) {
        return nullptr;
    }

    jint* pixels = env->GetIntArrayElements(result, nullptr);
    if (!pixels) {
        return nullptr;
    }

    const uint8_t* src = rgb;
    for (int i = 0; i < pixelCount; ++i) {
        pixels[i] = static_cast<jint>(
                0xFF000000u |
                (static_cast<uint32_t>(src[0]) << 16) |
                (static_cast<uint32_t>(src[1]) << 8) |
                static_cast<uint32_t>(src[2]));
        src += channel;
    }

    env->ReleaseIntArrayElements(result, pixels, 0);
    return result;
}

jobjectArray convert_sd_frames_to_java(JNIEnv* env, sd_image_t* frames, int numFrames) {
    jclass byteArrayClass = env->FindClass("[B");
    if (!byteArrayClass) {
        free_sd_generated_frames(frames, numFrames);
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(numFrames, byteArrayClass, nullptr);
    env->DeleteLocalRef(byteArrayClass);
    if (!result) {
        free_sd_generated_frames(frames, numFrames);
        throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate video frame array");
        return nullptr;
    }

    for (int i = 0; i < numFrames; ++i) {
        if (!frames[i].data) {
            free_sd_generated_frames(frames, numFrames);
            throwJavaException(env, "java/lang/IllegalStateException", "Missing frame data");
            return nullptr;
        }

        const size_t byteCount =
                static_cast<size_t>(frames[i].width) * frames[i].height * frames[i].channel;
        jbyteArray frameBytes = env->NewByteArray(static_cast<jsize>(byteCount));
        if (!frameBytes) {
            free_sd_generated_frames(frames, numFrames);
            throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate frame buffer");
            return nullptr;
        }

        env->SetByteArrayRegion(
                frameBytes,
                0,
                static_cast<jsize>(byteCount),
                reinterpret_cast<jbyte*>(frames[i].data));
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(frameBytes);
            free_sd_generated_frames(frames, numFrames);
            return nullptr;
        }

        env->SetObjectArrayElement(result, i, frameBytes);
        env->DeleteLocalRef(frameBytes);
        if (env->ExceptionCheck()) {
            free_sd_generated_frames(frames, numFrames);
            return nullptr;
        }
    }

    free_sd_generated_frames(frames, numFrames);
    return result;
}

void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    llmedge_throw_java_exception(env, className, message);
}

void clearProgressCallback(JNIEnv* env, SdHandle* handle) {
    if (!handle) {
        return;
    }
    llmedge_clear_global_ref(env, handle->progressCallbackGlobalRef);
    handle->progressMethodID = nullptr;
    handle->currentFrame = 0;
    handle->totalFrames = 0;
    handle->stepsPerFrame = 0;
    handle->totalSteps = 0;
    handle->cancellationRequested.store(false);
}

void sd_video_progress_wrapper(int step, int steps, float time, void* data) {
    auto* handle = static_cast<SdHandle*>(data);
    if (!handle) {
        return;
    }

    if (handle->cancellationRequested.load()) {
        throw std::runtime_error("Generation cancelled");
    }

    if (!handle->progressCallbackGlobalRef || !handle->jvm || !handle->progressMethodID) {
        return;
    }

    static thread_local int lastReportedStep = -1;
    const int totalSteps = steps > 0 ? steps : (handle->totalSteps > 0 ? handle->totalSteps : 1);
    if (totalSteps > 20) {
        const int pct = (step * 100) / totalSteps;
        const int lastPct = (lastReportedStep * 100) / totalSteps;
        if (pct / 5 == lastPct / 5 && step != 0 && step != totalSteps) {
            return;
        }
    }
    lastReportedStep = step;

    JNIEnv* env = jni_thread_cache_get_env();
    if (!env) {
        return;
    }

    const int totalFrames = handle->totalFrames > 0 ? handle->totalFrames : 1;
    int inferredFrame = 0;
    if (totalFrames > 1 && totalSteps > 0) {
        inferredFrame = (step * totalFrames) / totalSteps;
        if (inferredFrame < 0) {
            inferredFrame = 0;
        }
        if (inferredFrame >= totalFrames) {
            inferredFrame = totalFrames - 1;
        }
    }
    handle->currentFrame = inferredFrame;

    env->CallVoidMethod(
            handle->progressCallbackGlobalRef,
            handle->progressMethodID,
            static_cast<jint>(std::min(step, totalSteps)),
            static_cast<jint>(totalSteps),
            static_cast<jint>(handle->currentFrame),
            static_cast<jint>(handle->totalFrames),
            static_cast<jfloat>(time));

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}
