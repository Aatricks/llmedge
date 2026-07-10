# Bark.cpp JNI wrapper build (Text-to-Speech)
# Build bark, encodec, and their ggml sources directly without add_subdirectory
# to avoid target name conflicts with stable-diffusion.cpp's ggml.

option(LLMEDGE_BARK_AGGRESSIVE_ARM_OPT "Enable aggressive ARM tuning flags for Bark (may reduce correctness/compatibility)" OFF)

get_filename_component(BARK_DIR "${LLMEDGE_CPP_ROOT}/../../../../bark.cpp" ABSOLUTE)

if (NOT EXISTS "${BARK_DIR}/bark.h")
        message(FATAL_ERROR "bark.cpp headers not found at ${BARK_DIR}. Please ensure bark.cpp is cloned.")
endif()

set(BARK_ENCODEC_DIR ${BARK_DIR}/encodec.cpp)
set(BARK_GGML_DIR ${BARK_ENCODEC_DIR}/ggml)

set(BARK_GGML_SOURCES
        ${BARK_GGML_DIR}/src/ggml.c
        ${BARK_GGML_DIR}/src/ggml-alloc.c
        ${BARK_GGML_DIR}/src/ggml-backend.cpp
        ${BARK_GGML_DIR}/src/ggml-quants.c
        ${BARK_GGML_DIR}/src/ggml-aarch64.c
)

set(ENCODEC_SOURCES
        ${BARK_ENCODEC_DIR}/encodec.cpp
        ${BARK_ENCODEC_DIR}/ops.cpp
)

set(BARK_SOURCES
        ${BARK_DIR}/bark.cpp
)

set(BARK_JNI_ALL_SOURCES
        ${BARK_GGML_SOURCES}
        ${ENCODEC_SOURCES}
        ${BARK_SOURCES}
        ${LLMEDGE_CPP_ROOT}/bark_jni.cpp
        ${LLMEDGE_CPP_ROOT}/jni_thread_cache.cpp
)

add_library(${LLMEDGE_TARGET_BARK_JNI} SHARED ${BARK_JNI_ALL_SOURCES})

target_include_directories(${LLMEDGE_TARGET_BARK_JNI}
        PRIVATE
        ${BARK_DIR}
        ${BARK_ENCODEC_DIR}
        ${BARK_GGML_DIR}/include
        ${BARK_GGML_DIR}/src
)

target_compile_definitions(${LLMEDGE_TARGET_BARK_JNI}
        PRIVATE
        GGML_COMMIT=""
        GGML_VERSION=""
        EXPORTING_BARK
        GGML_USE_CPU
)

target_compile_features(${LLMEDGE_TARGET_BARK_JNI} PUBLIC c_std_11 cxx_std_17)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        target_compile_options(${LLMEDGE_TARGET_BARK_JNI} PRIVATE -march=armv8-a)

        if (LLMEDGE_BARK_AGGRESSIVE_ARM_OPT)
                target_compile_options(${LLMEDGE_TARGET_BARK_JNI} PRIVATE
                        -march=armv8.4-a+dotprod+fp16
                        -mtune=cortex-a78
                        -ffp-contract=fast
                        -fno-signed-zeros
                        -fno-trapping-math
                        -freciprocal-math
                )
        endif()
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
        target_compile_options(${LLMEDGE_TARGET_BARK_JNI} PRIVATE -mfpu=neon-vfpv4)
endif()

target_compile_options(${LLMEDGE_TARGET_BARK_JNI} PUBLIC
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -ffunction-sections
        -fdata-sections
        -O3
        -funroll-loops
)

# No OpenMP here: see llmedge-whisper.cmake — a second static libomp in the
# process aborts in libomp's duplicate-runtime check.
target_link_libraries(${LLMEDGE_TARGET_BARK_JNI}
        android log
)

target_link_options(${LLMEDGE_TARGET_BARK_JNI} PRIVATE -Wl,--gc-sections -flto -Wl,--exclude-libs,ALL)

message(STATUS "Bark.cpp JNI wrapper configured (direct source build)")
