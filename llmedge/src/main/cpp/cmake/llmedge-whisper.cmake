# Whisper.cpp JNI wrapper build (Speech-to-Text)
# Build whisper with its OWN ggml sources compiled directly into whisper_jni
# to avoid target name conflicts and ensure API compatibility.

include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-mods.cmake")

get_filename_component(WHISPER_DIR_UPSTREAM "${LLMEDGE_CPP_ROOT}/../../../../whisper.cpp" ABSOLUTE)
get_filename_component(LLMEDGE_REPO_ROOT "${LLMEDGE_CPP_ROOT}/../../../.." ABSOLUTE)
set(LLMEDGE_MODS_DIR "${LLMEDGE_REPO_ROOT}/mods")
llmedge_prepare_patch_modded_tree(
        NAME "whisper.cpp"
        SOURCE_ROOT "${WHISPER_DIR_UPSTREAM}"
        MODS_ROOT "${LLMEDGE_MODS_DIR}/whisper.cpp"
        USE_MODS_VAR "LLMEDGE_WHISPER_USE_MODS"
        OUT_VAR WHISPER_DIR
)
set(WHISPER_GGML_DIR ${WHISPER_DIR}/ggml)

if (NOT EXISTS "${WHISPER_DIR}/include/whisper.h")
        message(FATAL_ERROR "whisper.cpp headers not found at ${WHISPER_DIR}. Please ensure whisper.cpp is cloned.")
endif()

set(WHISPER_GGML_SOURCES
        ${WHISPER_GGML_DIR}/src/ggml.c
        ${WHISPER_GGML_DIR}/src/ggml-alloc.c
        ${WHISPER_GGML_DIR}/src/ggml-backend.cpp
        ${WHISPER_GGML_DIR}/src/ggml-backend-reg.cpp
        ${WHISPER_GGML_DIR}/src/ggml-backend-meta.cpp
        ${WHISPER_GGML_DIR}/src/ggml-opt.cpp
        ${WHISPER_GGML_DIR}/src/ggml-quants.c
        ${WHISPER_GGML_DIR}/src/ggml-threading.cpp
        ${WHISPER_GGML_DIR}/src/gguf.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/ggml-cpu.c
        ${WHISPER_GGML_DIR}/src/ggml-cpu/ggml-cpu.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/ops.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/vec.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/quants.c
        ${WHISPER_GGML_DIR}/src/ggml-cpu/traits.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/unary-ops.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/binary-ops.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/repack.cpp
)

if (ANDROID_ABI MATCHES "^(arm64-v8a|armeabi-v7a)$")
        list(APPEND WHISPER_GGML_SOURCES
                ${WHISPER_GGML_DIR}/src/ggml-cpu/arch/arm/quants.c
                ${WHISPER_GGML_DIR}/src/ggml-cpu/arch/arm/repack.cpp
                ${WHISPER_GGML_DIR}/src/ggml-cpu/arch/arm/cpu-feats.cpp
        )
elseif (ANDROID_ABI MATCHES "^(x86_64|x86)$")
        list(APPEND WHISPER_GGML_SOURCES
                ${WHISPER_GGML_DIR}/src/ggml-cpu/arch/x86/quants.c
                ${WHISPER_GGML_DIR}/src/ggml-cpu/arch/x86/repack.cpp
                ${WHISPER_GGML_DIR}/src/ggml-cpu/arch/x86/cpu-feats.cpp
        )
endif()

set(WHISPER_SOURCES
        ${WHISPER_DIR}/src/whisper.cpp
        ${LLMEDGE_CPP_ROOT}/whisper_jni_common.cpp
        ${LLMEDGE_CPP_ROOT}/whisper_jni.cpp
        ${LLMEDGE_CPP_ROOT}/jni_thread_cache.cpp
)

set(WHISPER_JNI_ALL_SOURCES
        ${WHISPER_GGML_SOURCES}
        ${WHISPER_SOURCES}
)

add_library(${LLMEDGE_TARGET_WHISPER_JNI} SHARED ${WHISPER_JNI_ALL_SOURCES})

target_compile_definitions(${LLMEDGE_TARGET_WHISPER_JNI} PUBLIC GGML_USE_CPU)
target_compile_definitions(${LLMEDGE_TARGET_WHISPER_JNI} PRIVATE
        WHISPER_VERSION="1.0.0"
        GGML_VERSION="0.9.4"
        GGML_COMMIT="unknown"
        GGML_USE_OPENMP
)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        target_compile_options(${LLMEDGE_TARGET_WHISPER_JNI} PRIVATE -march=armv8.2-a+fp16)
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
        target_compile_options(${LLMEDGE_TARGET_WHISPER_JNI} PRIVATE -mfpu=neon-vfpv4)
endif()

target_include_directories(${LLMEDGE_TARGET_WHISPER_JNI}
        PRIVATE
        ${WHISPER_DIR}/include
        ${WHISPER_DIR}/src
        ${WHISPER_GGML_DIR}/include
        ${WHISPER_GGML_DIR}/src
        ${WHISPER_GGML_DIR}/src/ggml-cpu
        ${WHISPER_GGML_DIR}/src/ggml-cpu/arch
)

target_compile_features(${LLMEDGE_TARGET_WHISPER_JNI} PUBLIC c_std_11 cxx_std_17)

target_compile_options(${LLMEDGE_TARGET_WHISPER_JNI} PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections -O3 -fopenmp)

if (LLMEDGE_OPENCL_ENABLED)
        llmedge_enable_android_opencl(${LLMEDGE_TARGET_WHISPER_JNI} "${WHISPER_GGML_DIR}")
endif()

target_link_libraries(${LLMEDGE_TARGET_WHISPER_JNI}
        android log
        -fopenmp -static-openmp
)

target_link_options(${LLMEDGE_TARGET_WHISPER_JNI} PRIVATE -Wl,--gc-sections -flto -Wl,--exclude-libs,ALL)

message(STATUS "Whisper.cpp JNI wrapper configured (direct source build with bundled ggml, OpenMP enabled)")
