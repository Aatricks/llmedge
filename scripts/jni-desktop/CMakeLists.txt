cmake_minimum_required(VERSION 3.10)
project(sdcpp_desktop)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(LLMEDGE_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../llmedge/src/main/cpp" ABSOLUTE)
include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-mods.cmake")
include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-targets.cmake")

# Point to stable-diffusion.cpp root.
set(SD_ROOT_OVERRIDE "" CACHE PATH "Override path to stable-diffusion.cpp root")
if (SD_ROOT_OVERRIDE)
    get_filename_component(SD_ROOT "${SD_ROOT_OVERRIDE}" ABSOLUTE)
else()
    get_filename_component(SD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../stable-diffusion.cpp" ABSOLUTE)
endif()
get_filename_component(LLMEDGE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(LLMEDGE_MODS_DIR "${LLMEDGE_REPO_ROOT}/mods")

# Options to control what to build
option(BUILD_SDCPP "Build stable-diffusion.cpp JNI" ON)
option(BUILD_SMOLLM "Build SmolLM (llama.cpp) JNI" OFF)
option(BUILD_BARK "Build Bark JNI" OFF)

# Find JNI
find_package(JNI REQUIRED)

# ------------------------------------------------------------
# Stable Diffusion JNI
# ------------------------------------------------------------
if(BUILD_SDCPP)
    message(STATUS "Using stable-diffusion.cpp root: ${SD_ROOT}")

    # Configure stable-diffusion.cpp options
    set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    if(NOT DEFINED SD_VULKAN)
        set(SD_VULKAN ON CACHE BOOL "" FORCE)
    endif()

    # Add stable-diffusion.cpp
    add_subdirectory(${SD_ROOT} stable-diffusion)

    # Build sdcpp shared library
    add_library(${LLMEDGE_TARGET_SDCPP} SHARED
        ${LLMEDGE_CPP_ROOT}/sdcpp_jni.cpp
    )

    target_include_directories(${LLMEDGE_TARGET_SDCPP} PRIVATE
        ${SD_ROOT}
        ${SD_ROOT}/src
        ${SD_ROOT}/thirdparty
        ${JNI_INCLUDE_DIRS}
    )

    target_link_libraries(${LLMEDGE_TARGET_SDCPP} PRIVATE
        stable-diffusion
        ${JNI_LIBRARIES}
    )

    if(SD_VULKAN)
        target_compile_definitions(${LLMEDGE_TARGET_SDCPP} PRIVATE SD_USE_VULKAN)
        target_link_libraries(${LLMEDGE_TARGET_SDCPP} PRIVATE vulkan)
        if(TARGET ggml-vulkan)
            target_link_libraries(${LLMEDGE_TARGET_SDCPP} PRIVATE ggml-vulkan)
        endif()
    endif()

    if(WAN_SUPPORT)
        target_compile_definitions(${LLMEDGE_TARGET_SDCPP} PRIVATE WAN_SUPPORT=1)
    endif()
endif()

# ------------------------------------------------------------
# SmolLM (llama.cpp) JNI
# ------------------------------------------------------------
if(BUILD_SMOLLM)
    get_filename_component(LLAMA_ROOT_UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp" ABSOLUTE)
    llmedge_prepare_patch_modded_tree(
        NAME "llama.cpp"
        SOURCE_ROOT "${LLAMA_ROOT_UPSTREAM}"
        MODS_ROOT "${LLMEDGE_MODS_DIR}/llama.cpp"
        USE_MODS_VAR "LLMEDGE_LLAMA_USE_MODS"
        OUT_VAR LLAMA_ROOT
    )
    message(STATUS "Using llama.cpp root: ${LLAMA_ROOT}")
    set(LLMEDGE_LLAMA_USE_MEMORY_API OFF CACHE BOOL "Use llama_memory_* APIs instead of llama_kv_cache_* compatibility APIs" FORCE)

    # Configure llama.cpp options
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
    set(LLAMA_LLGUIDANCE OFF CACHE BOOL "" FORCE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        # ik_llama.cpp IQK kernels only compile their real x86 path when the build enables
        # AVX2-or-better CPU features. Without this, Linux host builds fall back to stubbed
        # implementations that abort at runtime for Q8_KV / IQ*_K* inference.
        set(GGML_NATIVE ON CACHE BOOL "" FORCE)
        set(GGML_AVX2 ON CACHE BOOL "" FORCE)
        set(GGML_FMA ON CACHE BOOL "" FORCE)
        set(GGML_F16C ON CACHE BOOL "" FORCE)
    endif()
    if(NOT DEFINED GGML_VULKAN)
        set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    endif()

    # Add llama.cpp
    add_subdirectory(${LLAMA_ROOT} llama.cpp)
    add_subdirectory(${LLAMA_ROOT}/examples/mtmd llama.cpp-mtmd)
    add_subdirectory(${LLAMA_ROOT}/examples/quantize llama.cpp-quantize)

    # Build smollm shared library
    add_library(${LLMEDGE_TARGET_SMOLLM} SHARED
        ${LLMEDGE_CPP_ROOT}/smollm.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_load.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_completion.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_embeddings.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_state.cpp
        ${LLMEDGE_CPP_ROOT}/LLMInference.cpp
        ${LLMEDGE_CPP_ROOT}/gguf_reader_internal.cpp
        ${LLMEDGE_CPP_ROOT}/GGUFReader.cpp
    )

    target_include_directories(${LLMEDGE_TARGET_SMOLLM} PRIVATE
        ${LLAMA_ROOT}/include
        ${LLAMA_ROOT}/common
        ${LLAMA_ROOT}/examples/mtmd
        ${LLAMA_ROOT}/ggml/include
        ${LLAMA_ROOT}/ggml/src
        ${LLAMA_ROOT} # For some internal headers if needed
        ${JNI_INCLUDE_DIRS}
    )

    target_compile_definitions(${LLMEDGE_TARGET_SMOLLM} PRIVATE
        LLMEDGE_LLAMA_USE_MEMORY_API=$<BOOL:${LLMEDGE_LLAMA_USE_MEMORY_API}>
    )

    target_link_libraries(${LLMEDGE_TARGET_SMOLLM} PRIVATE
        llama
        common
        mtmd
        ${JNI_LIBRARIES}
    )
endif()

# ------------------------------------------------------------
# Bark JNI wrapper for desktop testing
# ------------------------------------------------------------

if(BUILD_BARK)
    get_filename_component(BARK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../bark.cpp" ABSOLUTE)
    if(EXISTS "${BARK_ROOT}/bark.h")
        set(BARK_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(ENCODEC_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

        add_subdirectory(${BARK_ROOT} bark.cpp)

        add_library(${LLMEDGE_TARGET_BARK_JNI} SHARED
            ${LLMEDGE_CPP_ROOT}/bark_jni.cpp
        )

        target_include_directories(${LLMEDGE_TARGET_BARK_JNI} PRIVATE
            ${BARK_ROOT}
            ${BARK_ROOT}/encodec.cpp
            ${BARK_ROOT}/encodec.cpp/ggml/include
            ${JNI_INCLUDE_DIRS}
        )

        target_link_libraries(${LLMEDGE_TARGET_BARK_JNI} PRIVATE
            bark
            encodec
            ggml
            ${JNI_LIBRARIES}
            pthread
            m
        )

        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${LLMEDGE_TARGET_BARK_JNI} PRIVATE OpenMP::OpenMP_CXX)
        endif()

        target_compile_options(${LLMEDGE_TARGET_BARK_JNI} PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden)

        message(STATUS "Bark desktop JNI configured")
    else()
        message(WARNING "bark.cpp not found at ${BARK_ROOT}, skipping bark_jni")
    endif()
endif()

# ------------------------------------------------------------
# Whisper.cpp JNI wrapper for desktop testing
# ------------------------------------------------------------

option(WHISPER_DESKTOP_JNI "Build whisper JNI for desktop" OFF)

if(WHISPER_DESKTOP_JNI)
    get_filename_component(WHISPER_ROOT_UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/../../whisper.cpp" ABSOLUTE)
    llmedge_prepare_patch_modded_tree(
        NAME "whisper.cpp"
        SOURCE_ROOT "${WHISPER_ROOT_UPSTREAM}"
        MODS_ROOT "${LLMEDGE_MODS_DIR}/whisper.cpp"
        USE_MODS_VAR "LLMEDGE_WHISPER_USE_MODS"
        OUT_VAR WHISPER_ROOT
    )

    if(EXISTS "${WHISPER_ROOT}/include/whisper.h")
        # Configure whisper.cpp build options
        set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
        set(WHISPER_CURL OFF CACHE BOOL "" FORCE)

        add_subdirectory(${WHISPER_ROOT} whisper.cpp)

        add_library(${LLMEDGE_TARGET_WHISPER_JNI} SHARED
            ${LLMEDGE_CPP_ROOT}/whisper_jni_common.cpp
            ${LLMEDGE_CPP_ROOT}/whisper_jni.cpp
        )

        target_include_directories(${LLMEDGE_TARGET_WHISPER_JNI} PRIVATE
            ${WHISPER_ROOT}/include
            ${WHISPER_ROOT}/ggml/include
            ${JNI_INCLUDE_DIRS}
        )

        target_link_libraries(${LLMEDGE_TARGET_WHISPER_JNI} PRIVATE
            whisper
            ${JNI_LIBRARIES}
        )

        message(STATUS "Whisper desktop JNI configured")
    else()
        message(WARNING "whisper.cpp not found at ${WHISPER_ROOT}, skipping whisper_jni")
    endif()
endif()
