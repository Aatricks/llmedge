#pragma once

#include <jni.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <regex>
#include <string>
#include <vector>

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

// sd_tensor_raw_t / sd_condition_raw_t are now defined in stable-diffusion.h (llmedge Lever 1).

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

struct SdResolvedPromptLoras {
    std::string prompt;
    std::string negativePrompt;
    std::vector<std::string> ownedPaths;
    std::vector<sd_lora_t> loras;
};

static inline void extract_prompt_loras(
        std::string* text,
        const std::string& lora_model_dir,
        std::map<std::string, float>* low_noise_loras,
        std::map<std::string, float>* high_noise_loras) {
    if (!text || lora_model_dir.empty() || !low_noise_loras || !high_noise_loras) {
        return;
    }

    static const std::regex re(R"(<lora:([^:>]+):([^>]+)>)");
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt"};

    std::string remaining = *text;
    std::smatch match;

    while (std::regex_search(remaining, match, re)) {
        std::string raw_path = match[1].str();
        float multiplier = 0.0f;
        try {
            multiplier = std::stof(match[2].str());
        } catch (...) {
            remaining = match.suffix().str();
            *text = std::regex_replace(*text, re, "", std::regex_constants::format_first_only);
            continue;
        }

        bool is_high_noise = false;
        static const std::string high_noise_prefix = "|high_noise|";
        if (raw_path.rfind(high_noise_prefix, 0) == 0) {
            raw_path.erase(0, high_noise_prefix.size());
            is_high_noise = true;
        }

        std::filesystem::path resolved =
                std::filesystem::path(raw_path).is_absolute()
                        ? std::filesystem::path(raw_path)
                        : std::filesystem::path(lora_model_dir) / raw_path;

        if (!std::filesystem::exists(resolved)) {
            bool found_with_extension = false;
            for (const std::string& extension : valid_ext) {
                std::filesystem::path candidate = resolved;
                candidate += extension;
                if (std::filesystem::exists(candidate)) {
                    resolved = candidate;
                    found_with_extension = true;
                    break;
                }
            }
            if (!found_with_extension) {
                ALOGW("Unable to resolve LoRA '%s' relative to '%s'", raw_path.c_str(), lora_model_dir.c_str());
                remaining = match.suffix().str();
                *text = std::regex_replace(*text, re, "", std::regex_constants::format_first_only);
                continue;
            }
        }

        const std::string normalized = resolved.lexically_normal().string();
        if (is_high_noise) {
            (*high_noise_loras)[normalized] += multiplier;
        } else {
            (*low_noise_loras)[normalized] += multiplier;
        }

        *text = std::regex_replace(*text, re, "", std::regex_constants::format_first_only);
        remaining = match.suffix().str();
    }
}

static inline SdResolvedPromptLoras resolve_prompt_loras(
        const char* raw_prompt,
        const char* raw_negative_prompt,
        const std::string& lora_model_dir) {
    SdResolvedPromptLoras resolved;
    resolved.prompt = raw_prompt ? raw_prompt : "";
    resolved.negativePrompt = raw_negative_prompt ? raw_negative_prompt : "";

    if (lora_model_dir.empty()) {
        return resolved;
    }

    std::map<std::string, float> low_noise_loras;
    std::map<std::string, float> high_noise_loras;
    extract_prompt_loras(&resolved.prompt, lora_model_dir, &low_noise_loras, &high_noise_loras);
    extract_prompt_loras(&resolved.negativePrompt, lora_model_dir, &low_noise_loras, &high_noise_loras);

    const size_t total_loras = low_noise_loras.size() + high_noise_loras.size();
    resolved.ownedPaths.reserve(total_loras);
    resolved.loras.reserve(total_loras);

    for (const auto& entry : low_noise_loras) {
        resolved.ownedPaths.push_back(entry.first);
        sd_lora_t item{};
        item.is_high_noise = false;
        item.multiplier = entry.second;
        item.path = resolved.ownedPaths.back().c_str();
        resolved.loras.push_back(item);
    }
    for (const auto& entry : high_noise_loras) {
        resolved.ownedPaths.push_back(entry.first);
        sd_lora_t item{};
        item.is_high_noise = true;
        item.multiplier = entry.second;
        item.path = resolved.ownedPaths.back().c_str();
        resolved.loras.push_back(item);
    }

    return resolved;
}
