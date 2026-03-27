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
        ${WHISPER_GGML_DIR}/src/ggml-cpu/arch/arm/quants.c
)

set(WHISPER_SOURCES
        ${WHISPER_DIR}/src/whisper.cpp
        ${LLMEDGE_CPP_ROOT}/whisper_jni.cpp
)

set(WHISPER_JNI_ALL_SOURCES
        ${WHISPER_GGML_SOURCES}
        ${WHISPER_SOURCES}
)

add_library(whisper_jni SHARED ${WHISPER_JNI_ALL_SOURCES})

target_compile_definitions(whisper_jni PUBLIC GGML_USE_CPU)
target_compile_definitions(whisper_jni PRIVATE
        WHISPER_VERSION="1.0.0"
        GGML_VERSION="0.9.4"
        GGML_COMMIT="unknown"
        GGML_USE_OPENMP
)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        target_compile_options(whisper_jni PRIVATE -march=armv8.2-a+fp16)
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
        target_compile_options(whisper_jni PRIVATE -mfpu=neon-vfpv4)
endif()

target_include_directories(whisper_jni
        PRIVATE
        ${WHISPER_DIR}/include
        ${WHISPER_DIR}/src
        ${WHISPER_GGML_DIR}/include
        ${WHISPER_GGML_DIR}/src
        ${WHISPER_GGML_DIR}/src/ggml-cpu
        ${WHISPER_GGML_DIR}/src/ggml-cpu/arch
)

target_compile_features(whisper_jni PUBLIC c_std_11 cxx_std_17)

target_compile_options(whisper_jni PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections -O3 -fopenmp)

target_link_libraries(whisper_jni
        android log
        -fopenmp -static-openmp
)

target_link_options(whisper_jni PRIVATE -Wl,--gc-sections -flto -Wl,--exclude-libs,ALL)

message(STATUS "Whisper.cpp JNI wrapper configured (direct source build with bundled ggml, OpenMP enabled)")
