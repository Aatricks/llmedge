# Shared native target names used by Android and desktop CMake entrypoints.

set(LLMEDGE_TARGET_SMOLLM "smollm")
set(LLMEDGE_TARGET_SMOLLM_V7A "smollm_v7a")
set(LLMEDGE_TARGET_SMOLLM_V8_2_FP16_DOTPROD "smollm_v8_2_fp16_dotprod")
set(LLMEDGE_TARGET_SMOLLM_V8_4_FP16_DOTPROD_I8MM "smollm_v8_4_fp16_dotprod_i8mm")
set(LLMEDGE_TARGET_GGUF_READER "ggufreader")
set(LLMEDGE_TARGET_SDCPP "sdcpp")
set(LLMEDGE_TARGET_WHISPER_JNI "whisper_jni")
set(LLMEDGE_TARGET_BARK_JNI "bark_jni")

# Shared logical target groups consumed by desktop build/test scripts.
set(LLMEDGE_DESKTOP_TARGETS "smollm;whisper;sdcpp;bark")
set(LLMEDGE_CI_TARGETS "whisper;sdcpp;smollm")
