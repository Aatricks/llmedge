# SmolLM + GGUF reader JNI targets.

if (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library_armv7a("${LLMEDGE_TARGET_SMOLLM}" "-march=armv7-a" "-mfpu=neon-vfpv4" "-mfloat-abi=softfp")
    build_library_armv7a("${LLMEDGE_TARGET_SMOLLM_V7A}" "-march=armv7-a" "-mfpu=neon-vfpv4" "-mfloat-abi=softfp")
else()
    build_library_universal("${LLMEDGE_TARGET_SMOLLM}")
endif()
if (LLMEDGE_OPENCL_ENABLED)
    llmedge_enable_android_opencl("${LLMEDGE_TARGET_SMOLLM}" "${GGML_DIR}")
endif()
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    # Two tuned variants beyond the universal baseline (which is already armv8-a
    # with IQK enabled, so a separate -march=armv8-a target would be a duplicate).
    # The vendored IQK kernels gate on fp16+dotprod only — no SVE and no i8mm —
    # so SVE builds bought nothing, and i8mm (used by ggml's aarch64 repack
    # paths) ships in a single v8.4 build. Keeping the variant count at three
    # roughly third's the AAR's smollm payload.
    build_library_arm64("${LLMEDGE_TARGET_SMOLLM_V8_2_FP16_DOTPROD}" "-march=armv8.2-a+fp16+dotprod")
    if (LLMEDGE_OPENCL_ENABLED)
        llmedge_enable_android_opencl("${LLMEDGE_TARGET_SMOLLM_V8_2_FP16_DOTPROD}" "${GGML_DIR}")
    endif()
    build_library_arm64("${LLMEDGE_TARGET_SMOLLM_V8_4_FP16_DOTPROD_I8MM}" "-march=armv8.4-a+fp16+dotprod+i8mm")
    if (LLMEDGE_OPENCL_ENABLED)
        llmedge_enable_android_opencl("${LLMEDGE_TARGET_SMOLLM_V8_4_FP16_DOTPROD_I8MM}" "${GGML_DIR}")
    endif()
endif()

# library target for GGUFReader
set(TARGET_NAME_GGUF_READER ${LLMEDGE_TARGET_GGUF_READER})
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
if (NOT ${ANDROID_ABI} STREQUAL "armeabi-v7a")
    target_sources(${TARGET_NAME_GGUF_READER} PRIVATE ${LLMEDGE_IQK_BASE_SOURCES})
endif()
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
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
