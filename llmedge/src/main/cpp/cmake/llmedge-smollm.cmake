# SmolLM + GGUF reader JNI targets.

if (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library_armv7a("smollm" "-march=armv7-a" "-mfpu=neon-vfpv4" "-mfloat-abi=softfp")
    build_library_armv7a("smollm_v7a" "-march=armv7-a" "-mfpu=neon-vfpv4" "-mfloat-abi=softfp")
else()
    build_library_universal("smollm")
endif()
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library_arm64("smollm_v8" "-march=armv8-a")
    # Targets for Arm-v8.2a
    build_library_arm64("smollm_v8_2_fp16" "-march=armv8.2-a+fp16")
    build_library_arm64("smollm_v8_2_fp16_dotprod" "-march=armv8.2-a+fp16+dotprod")

    # Targets for Arm-v8.4a
    build_library_arm64("smollm_v8_4_fp16_dotprod" "-march=armv8.4-a+fp16+dotprod")
    build_library_arm64("smollm_v8_4_fp16_dotprod_sve" "-march=armv8.4-a+fp16+dotprod+sve")
    target_compile_definitions(smollm_v8_4_fp16_dotprod_sve PRIVATE GGML_F32_STEP=32)
    build_library_arm64("smollm_v8_4_fp16_dotprod_i8mm" "-march=armv8.4-a+fp16+dotprod+i8mm")
    build_library_arm64("smollm_v8_4_fp16_dotprod_i8mm_sve" "-march=armv8.4-a+fp16+dotprod+i8mm+sve")
    target_compile_definitions(smollm_v8_4_fp16_dotprod_i8mm_sve PRIVATE GGML_F32_STEP=32)
endif()

# library target for GGUFReader
set(TARGET_NAME_GGUF_READER ggufreader)
add_library(${TARGET_NAME_GGUF_READER} SHARED ${GGUF_READER_SOURCES})
target_compile_features(${TARGET_NAME_GGUF_READER} PUBLIC c_std_11 cxx_std_20)
target_include_directories(
        ${TARGET_NAME_GGUF_READER}
        PUBLIC
        ${GGML_DIR}/include
        ${GGML_DIR}/src
        ${GGML_DIR}/src/iqk
)
# Constants GGML_COMMIT and GGML_VERSION in ggml.c
# are supplied through llama.cpp's CMake script (that in turn gets them from git)
# As we are NOT using llama.cpp's CMake script, we need to define them here.
target_compile_definitions(
        ${TARGET_NAME_GGUF_READER}
        PRIVATE
        GGML_COMMIT=""
        GGML_VERSION=""
)
set(_gguf_reader_cpu_flags -DGGML_USE_CPU)
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_sources(${TARGET_NAME_GGUF_READER} PRIVATE ${LLMEDGE_IQK_BASE_SOURCES})
    target_compile_definitions(
            ${TARGET_NAME_GGUF_READER}
            PRIVATE
            GGML_USE_IQK_MULMAT
            GGML_IQK_FLASH_ATTENTION
    )
    list(APPEND _gguf_reader_cpu_flags -DGGML_USE_CPU_AARCH64)
endif()
target_compile_options(
        ${TARGET_NAME_GGUF_READER}
        PUBLIC
        ${_gguf_reader_cpu_flags}
        -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections -O3
)
target_link_options(
        ${TARGET_NAME_GGUF_READER}
        PRIVATE
        -Wl,--gc-sections -flto
        -Wl,--exclude-libs,ALL
)
