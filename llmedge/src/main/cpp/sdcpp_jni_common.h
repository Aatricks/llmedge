#pragma once

#include <jni.h>

#include "stable-diffusion.h"
#include "sd_jni_internal.h"

SD_JNI_INTERNAL void free_sd_generated_frames(sd_image_t* frames, int numFrames);
SD_JNI_INTERNAL jintArray rgb_to_argb_int_array(
        JNIEnv* env,
        const uint8_t* rgb,
        int width,
        int height,
        int channel);
SD_JNI_INTERNAL jobjectArray convert_sd_frames_to_java(JNIEnv* env, sd_image_t* frames, int numFrames);
