#pragma once

#include <jni.h>

#include <algorithm>
#include <cstdint>

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

    va_start(args, format);
    fprintf(stdout, "[%s] ", tag);
    vfprintf(stdout, format, args);
    fprintf(stdout, "\n");
    fflush(stdout);
    va_end(args);
    return 0;
}
#endif

#ifndef GGML_MAX_NAME
#define GGML_MAX_NAME 128
#endif

#include "stable-diffusion.h"
#include "sd_jni_internal.h"
#include "sdcpp_jni_common.h"

#define LOG_TAG "SmolSD"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

typedef struct {
    int ndims;
    int64_t ne[4];
    float* data;
} sd_tensor_raw_t;

typedef struct {
    sd_tensor_raw_t c_crossattn;
    sd_tensor_raw_t c_vector;
    sd_tensor_raw_t c_concat;
} sd_condition_raw_t;

extern "C" SD_API sd_condition_raw_t* sd_precompute_condition(
        sd_ctx_t* ctx,
        const char* prompt,
        int clip_skip,
        int width,
        int height,
        bool is_video);

extern "C" SD_API void sd_free_condition(sd_condition_raw_t* cond);

extern "C" SD_API sd_image_t* sd_generate_image_with_precomputed_condition(
        sd_ctx_t* sd_ctx,
        const sd_img_gen_params_t* params,
        const sd_condition_raw_t* cond,
        const sd_condition_raw_t* uncond);

extern "C" SD_API sd_image_t* sd_generate_video_with_precomputed_condition(
        sd_ctx_t* sd_ctx,
        const sd_vid_gen_params_t* params,
        const sd_condition_raw_t* cond,
        const sd_condition_raw_t* uncond,
        int* num_frames_out);

static inline int sd_get_num_physical_cores_safe() {
    const int32_t n = sd_get_num_physical_cores();
    return n > 0 ? static_cast<int>(n) : 1;
}

static inline bool map_sample_method_from_kotlin_id(int kotlin_id, enum sample_method_t* out) {
    if (!out) {
        return false;
    }
    switch (kotlin_id) {
        case 1:  *out = EULER_SAMPLE_METHOD; break;
        case 12: *out = EULER_A_SAMPLE_METHOD; break;
        case 2:  *out = HEUN_SAMPLE_METHOD; break;
        case 3:  *out = DPM2_SAMPLE_METHOD; break;
        case 4:  *out = DPMPP2S_A_SAMPLE_METHOD; break;
        case 5:  *out = DPMPP2M_SAMPLE_METHOD; break;
        case 6:  *out = DPMPP2Mv2_SAMPLE_METHOD; break;
        case 7:  *out = IPNDM_SAMPLE_METHOD; break;
        case 8:  *out = IPNDM_V_SAMPLE_METHOD; break;
        case 9:  *out = LCM_SAMPLE_METHOD; break;
        case 10: *out = DDIM_TRAILING_SAMPLE_METHOD; break;
        case 11: *out = TCD_SAMPLE_METHOD; break;
        default:
            return false;
    }
    return true;
}

static inline bool map_scheduler_from_kotlin_id(int kotlin_id, enum scheduler_t* out) {
    if (!out) {
        return false;
    }
    if (kotlin_id <= 0) {
        return false;
    }
    const int upstream = kotlin_id - 1;
    if (upstream < 0 || upstream >= static_cast<int>(SCHEDULER_COUNT)) {
        return false;
    }
    *out = static_cast<enum scheduler_t>(upstream);
    return true;
}

static inline void apply_easycache_compat(
        sd_cache_params_t* cache,
        bool enabled,
        float reuse_threshold,
        float start_percent,
        float end_percent) {
    if (!cache) {
        return;
    }
    cache->mode = enabled ? SD_CACHE_EASYCACHE : SD_CACHE_DISABLED;
    cache->reuse_threshold = reuse_threshold;
    cache->start_percent = start_percent;
    cache->end_percent = end_percent;
}
