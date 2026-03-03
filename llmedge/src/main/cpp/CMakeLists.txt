cmake_minimum_required(VERSION 3.22.1)
project("smollm")

# Resolve LLAMA_DIR to an absolute path
set(LLAMA_DIR_RELATIVE "../../../../llama.cpp")
get_filename_component(LLAMA_DIR ${LLAMA_DIR_RELATIVE} ABSOLUTE)

set(GGML_DIR ${LLAMA_DIR}/ggml)
set(COMMON_DIR ${LLAMA_DIR}/common)
set(VENDOR_DIR ${LLAMA_DIR}/vendor)
set(SMOLLM_SOURCES
        ${GGML_DIR}/src/ggml-alloc.c
        ${GGML_DIR}/src/ggml-backend.cpp
        ${GGML_DIR}/src/ggml-threading.cpp
        ${GGML_DIR}/src/ggml-quants.c
        ${GGML_DIR}/src/ggml-backend-reg.cpp
        ${GGML_DIR}/src/ggml-opt.cpp
        ${GGML_DIR}/src/ggml-cpu/arch/arm/quants.c
        ${GGML_DIR}/src/ggml-cpu/ops.cpp
        ${GGML_DIR}/src/ggml-cpu/vec.cpp
        ${GGML_DIR}/src/ggml-cpu/quants.c
        ${GGML_DIR}/src/ggml-cpu/traits.cpp
        ${GGML_DIR}/src/ggml-cpu/unary-ops.cpp
        ${GGML_DIR}/src/ggml-cpu/binary-ops.cpp
        ${GGML_DIR}/src/ggml-cpu/ggml-cpu.c
        ${GGML_DIR}/src/ggml-cpu/ggml-cpu.cpp
        ${GGML_DIR}/src/ggml.c
        ${GGML_DIR}/src/gguf.cpp

        ${LLAMA_DIR}/src/llama.cpp
        ${LLAMA_DIR}/src/llama-vocab.cpp
        ${LLAMA_DIR}/src/llama-grammar.cpp
        ${LLAMA_DIR}/src/llama-sampling.cpp
        ${LLAMA_DIR}/src/llama-context.cpp
        ${LLAMA_DIR}/src/llama-model.cpp
        ${LLAMA_DIR}/src/llama-model-loader.cpp
        ${LLAMA_DIR}/src/llama-impl.cpp
        ${LLAMA_DIR}/src/llama-io.cpp
        ${LLAMA_DIR}/src/llama-memory.cpp
        ${LLAMA_DIR}/src/llama-memory-recurrent.cpp
        ${LLAMA_DIR}/src/llama-memory-hybrid.cpp
        ${LLAMA_DIR}/src/llama-mmap.cpp
        ${LLAMA_DIR}/src/llama-hparams.cpp
        ${LLAMA_DIR}/src/llama-kv-cache-iswa.cpp
        ${LLAMA_DIR}/src/llama-kv-cache.cpp
        ${LLAMA_DIR}/src/llama-batch.cpp
        ${LLAMA_DIR}/src/llama-arch.cpp
        ${LLAMA_DIR}/src/llama-adapter.cpp
        ${LLAMA_DIR}/src/llama-chat.cpp
        ${LLAMA_DIR}/src/llama-graph.cpp
        ${LLAMA_DIR}/src/unicode.h
        ${LLAMA_DIR}/src/unicode.cpp
        ${LLAMA_DIR}/src/unicode-data.cpp
        # Model builders - if llama.cpp is using per-model files, include them
        # otherwise rely on llama-model.cpp which contains model builders in newer
        # versions of llama.cpp

        ${VENDOR_DIR}/nlohmann/json_fwd.hpp
        ${VENDOR_DIR}/nlohmann/json.hpp

        ${COMMON_DIR}/arg.cpp
        ${COMMON_DIR}/base64.hpp
        ${COMMON_DIR}/common.cpp
        ${COMMON_DIR}/console.cpp
        ${COMMON_DIR}/json-schema-to-grammar.cpp
        ${COMMON_DIR}/log.cpp
        ${COMMON_DIR}/ngram-cache.cpp
        ${COMMON_DIR}/sampling.cpp

        LLMInference.cpp
        smollm.cpp
        # libmtmd (multimodal projector) from llama.cpp
        ${LLAMA_DIR}/tools/mtmd/mtmd.cpp
        ${LLAMA_DIR}/tools/mtmd/mtmd-helper.cpp
        ${LLAMA_DIR}/tools/mtmd/mtmd-audio.cpp
        ${LLAMA_DIR}/tools/mtmd/clip.cpp
)

# If llama.cpp contains per-model source files and does *not* provide a consolidated
# `llama-model.cpp`, append them to SMOLLM_SOURCES. This avoids duplicate symbols when the
# consolidated model source is present in newer llama.cpp versions.
if(EXISTS "${LLAMA_DIR}/src/models" AND NOT EXISTS "${LLAMA_DIR}/src/llama-model.cpp")
        file(GLOB LLAMA_MODEL_SOURCES "${LLAMA_DIR}/src/models/*.cpp")
        if(LLAMA_MODEL_SOURCES)
                list(APPEND SMOLLM_SOURCES ${LLAMA_MODEL_SOURCES})
        endif()
endif()

# ------------------------------------------------------------
# Vulkan configuration (Global)
# ------------------------------------------------------------
option(GGML_VULKAN "Enable Vulkan support" ON)
set(VULKAN_ENABLED ${GGML_VULKAN})

if (VULKAN_ENABLED)
    # Provide Vulkan library and include path for Android NDK
    find_library(VULKAN_LIB vulkan REQUIRED)
    set(Vulkan_INCLUDE_DIR "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include")

    # Download Vulkan C++ headers (vulkan.hpp) which don't ship with Android NDK
    include(FetchContent)
    FetchContent_Declare(
        vulkan_hpp
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Hpp.git
        GIT_TAG        v1.3.275  # Must match Android NDK's Vulkan version
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(vulkan_hpp)

    # Configure Vulkan for both llama.cpp and stable-diffusion.cpp ggml-vulkan
    set(Vulkan_INCLUDE_DIRS "${Vulkan_INCLUDE_DIR};${vulkan_hpp_SOURCE_DIR}")
    set(Vulkan_LIBRARIES "${VULKAN_LIB}")
    set(Vulkan_FOUND ON)
    set(Vulkan_INCLUDE_DIR "${Vulkan_INCLUDE_DIR}" CACHE PATH "Vulkan include directory" FORCE)
    set(Vulkan_LIBRARY "${VULKAN_LIB}" CACHE FILEPATH "Vulkan library" FORCE)

    # Find glslc compiler for Vulkan shader compilation
    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc HINTS $ENV{VULKAN_SDK}/bin)
    message(STATUS "Vulkan_GLSLC_EXECUTABLE: ${Vulkan_GLSLC_EXECUTABLE}")
    if(Vulkan_GLSLC_EXECUTABLE)
        set(Vulkan_glslc_FOUND ON)
        set(Vulkan_glslc_EXECUTABLE "${Vulkan_GLSLC_EXECUTABLE}")
    endif()
endif()

set(GGUF_READER_SOURCES
        ${GGML_DIR}/src/ggml.c
        ${GGML_DIR}/src/ggml-alloc.c
        ${GGML_DIR}/src/ggml-backend.cpp
        ${GGML_DIR}/src/ggml-threading.cpp
        ${GGML_DIR}/src/ggml-quants.c
        ${GGML_DIR}/src/ggml-backend-reg.cpp
        ${GGML_DIR}/src/ggml-opt.cpp
        ${GGML_DIR}/src/ggml-cpu/ops.cpp
        ${GGML_DIR}/src/ggml-cpu/vec.cpp
        ${GGML_DIR}/src/ggml-cpu/unary-ops.cpp
        ${GGML_DIR}/src/ggml-cpu/binary-ops.cpp
        ${GGML_DIR}/src/ggml-cpu/ggml-cpu.c
        ${GGML_DIR}/src/ggml-cpu/ggml-cpu.cpp
        ${GGML_DIR}/src/gguf.cpp
        GGUFReader.cpp
)

add_compile_options("-ffile-prefix-map=${LLAMA_DIR}=.")
add_link_options("LINKER:--build-id=none")

# compiling for different CPU extensions for Arm64 (aarch64)
# See docs/build_arm_flags.md for more details

function(build_library target_name)
    add_library(
            ${target_name}
            SHARED
            ${SMOLLM_SOURCES}
    )
    target_include_directories(
            ${target_name}
            PUBLIC
            ${COMMON_DIR}
            ${GGML_DIR}/include
            ${GGML_DIR}/src
            ${GGML_DIR}/src/ggml-cpu
            ${LLAMA_DIR}/include
            ${LLAMA_DIR}/tools/mtmd
            ${VENDOR_DIR}
    )

    # Constants GGML_COMMIT and GGML_VERSION in ggml.c
    # are supplied through llama.cpp's CMake script (that in turn gets them from git)
    # As we are NOT using llama.cpp's CMake script, we need to define them here
    # The functions used to access these constants are unused, hence setting them
    # to empty strings does not affect any functionality.
    target_compile_definitions(
            ${target_name}
            PRIVATE
            GGML_COMMIT=""
            GGML_VERSION=""
    )

    # -fvisibility=hidden: hide all symbols by default
    # -fvisibility-inlines-hidden: hide all inline symbols by default
    target_compile_options(
            ${target_name}
            PUBLIC
            -fvisibility=hidden -fvisibility-inlines-hidden
    )
    # -ffunction-sections: place each function in its own section
    # -fdata-sections: place each data member in its own section
    target_compile_options(
            ${target_name}
            PUBLIC
            -ffunction-sections -fdata-sections
    )

    # Enable OpenMP for multi-threaded ggml matrix operations
    target_compile_definitions(${target_name} PRIVATE GGML_USE_OPENMP)
    target_compile_options(${target_name} PUBLIC -fopenmp)

    target_link_libraries(
            ${target_name}
            android log
            -fopenmp -static-openmp
    )

    # -Wl,--gc-sections: remove unused sections (garbage collection)
    # -flto: link-time optimization
    # -Wl,--exclude-libs,ALL: exclude all libraries
    target_link_options(
            ${target_name}
            PRIVATE
            -Wl,--gc-sections -flto
            -Wl,--exclude-libs,ALL
    )
endfunction()


function(build_library_arm64 target_name cpu_flags)
    build_library(${target_name})
    target_compile_options(
            ${target_name}
            PUBLIC
            -DGGML_USE_CPU -DGGML_USE_CPU_AARCH64 ${cpu_flags} -O3 -funroll-loops
    )
endfunction()

function(build_library_armv7a target_name cpu_flags fpu fpu_abi)
    build_library(${target_name})
    target_compile_options(
            ${target_name}
            PUBLIC
            -DGGML_USE_CPU ${cpu_flags} ${fpu} ${fpu_abi} -O3 -funroll-loops
    )
endfunction()

function(build_library_universal target_name)
    build_library(${target_name})
    target_compile_options(
            ${target_name}
            PUBLIC
            -DGGML_USE_CPU -DGGML_USE_CPU_AARCH64 -O3 -funroll-loops
    )
endfunction()

build_library_universal("smollm")
if (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library_armv7a("smollm_v7a" "-march=armv7-a" "-mfpu=neon-vfpv4" "-mfloat-abi=softfp")
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
target_include_directories(
        ${TARGET_NAME_GGUF_READER}
        PUBLIC
        ${GGML_DIR}/include
        ${GGML_DIR}/src
        ${GGML_DIR}/src/ggml-cpu
)
# Constants GGML_COMMIT and GGML_VERSION in ggml.c
# are supplied through llama.cpp's CMake script (that in turn gets them from git)
# As we are NOT using llama.cpp's CMake script, we need to define them here
# The functions used to access these constants are unused, hence setting them
# to empty strings does not affect any functionality.
target_compile_definitions(
        ${TARGET_NAME_GGUF_READER}
        PRIVATE
        GGML_COMMIT=""
        GGML_VERSION=""
)
target_compile_options(
        ${TARGET_NAME_GGUF_READER}
        PUBLIC
        -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections -O3
)
target_link_options(
        ${TARGET_NAME_GGUF_READER}
        PRIVATE
        -Wl,--gc-sections -flto
        -Wl,--exclude-libs,ALL
)

# ------------------------------------------------------------
# Stable Diffusion JNI wrapper build (Vulkan enabled)
# ------------------------------------------------------------

# Resolve SD_DIR to an absolute path (upstream submodule)
set(SD_DIR_RELATIVE "../../../../stable-diffusion.cpp")
get_filename_component(SD_DIR_UPSTREAM ${SD_DIR_RELATIVE} ABSOLUTE)

# Optional: build stable-diffusion.cpp from a copied tree with overlays from the repo's
# `mods/` directory. This keeps the git submodule working tree pristine (no `-dirty`).
#
# Controls:
#   -DSD_ROOT_OVERRIDE=/abs/path      : use that stable-diffusion.cpp root directly
#   -DLLMEDGE_SDCPP_USE_MODS=ON       : enable copy+overlay
#   -DLLMEDGE_SDCPP_MODS_FILES=...    : semicolon- or comma-separated list of overlay files
set(SD_ROOT_OVERRIDE "" CACHE PATH "Override path to stable-diffusion.cpp root")

get_filename_component(LLMEDGE_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(LLMEDGE_MODS_DIR "${LLMEDGE_REPO_ROOT}/mods")

set(_LLMEDGE_SDCPP_USE_MODS_DEFAULT OFF)
if (EXISTS "${LLMEDGE_MODS_DIR}/stable-diffusion.cpp" OR EXISTS "${LLMEDGE_MODS_DIR}/stable-diffusion.h")
        set(_LLMEDGE_SDCPP_USE_MODS_DEFAULT ON)
endif()
option(LLMEDGE_SDCPP_USE_MODS "Build stable-diffusion.cpp from a copied tree with mods/ overlays (keeps submodule clean)" ${_LLMEDGE_SDCPP_USE_MODS_DEFAULT})

# Default overlays: only include wan.hpp when it exists in mods/.
set(_LLMEDGE_SDCPP_MODS_FILES_DEFAULT
    stable-diffusion.cpp
    stable-diffusion.h
    ggml_extend.hpp
    util.cpp
    ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
)
if (EXISTS "${LLMEDGE_MODS_DIR}/wan.hpp")
        set(_LLMEDGE_SDCPP_MODS_FILES_DEFAULT "wan.hpp;${_LLMEDGE_SDCPP_MODS_FILES_DEFAULT}")
endif()
set(LLMEDGE_SDCPP_MODS_FILES "${_LLMEDGE_SDCPP_MODS_FILES_DEFAULT}" CACHE STRING "Overlay files from mods/ onto the copied stable-diffusion.cpp tree (semicolon- or comma-separated)")

set(SD_DIR "${SD_DIR_UPSTREAM}")
if (SD_ROOT_OVERRIDE)
        get_filename_component(SD_DIR "${SD_ROOT_OVERRIDE}" ABSOLUTE)
elseif (LLMEDGE_SDCPP_USE_MODS OR (DEFINED ENV{LLMEDGE_SDCPP_USE_MODS} AND "$ENV{LLMEDGE_SDCPP_USE_MODS}" STREQUAL "1"))
        set(PATCHED_SD_ROOT "${CMAKE_CURRENT_BINARY_DIR}/patched-sd-src")
        message(STATUS "LLMEDGE_SDCPP_USE_MODS enabled: copying stable-diffusion.cpp sources to ${PATCHED_SD_ROOT} and overlaying mods")
        execute_process(COMMAND ${CMAKE_COMMAND} -E rm -rf "${PATCHED_SD_ROOT}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${PATCHED_SD_ROOT}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory "${SD_DIR_UPSTREAM}" "${PATCHED_SD_ROOT}")

        # Remove any stale ggml-vulkan shader-generator build/prefix directories that
        # may have been created by a previous CMake configure which referenced a
        # different stable-diffusion.cpp source directory. If left in place, CMake
        # will error out with a source-directory mismatch when the external
        # project attempts to (re)configure the host-side shader generator.
        file(GLOB _vulkan_gen_dirs "${CMAKE_CURRENT_BINARY_DIR}/stable-diffusion.cpp/ggml/src/ggml-vulkan/*-gen*")
        foreach(_d IN LISTS _vulkan_gen_dirs)
                if (EXISTS "${_d}")
                        message(STATUS "LLMEDGE_SDCPP_USE_MODS: removing stale Vulkan gen dir ${_d}")
                        execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${_d}")
                endif()
        endforeach()

        set(_mods_files "${LLMEDGE_SDCPP_MODS_FILES}")
        string(REPLACE "," ";" _mods_files "${_mods_files}")
        foreach(f IN LISTS _mods_files)
                string(STRIP "${f}" f_trimmed)
                if (f_trimmed STREQUAL "")
                        continue()
                endif()
                if (EXISTS "${LLMEDGE_MODS_DIR}/${f_trimmed}")
                        execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${LLMEDGE_MODS_DIR}/${f_trimmed}" "${PATCHED_SD_ROOT}/${f_trimmed}")
                else()
                        message(STATUS "LLMEDGE_SDCPP_USE_MODS: mods/${f_trimmed} not found; skipping")
                endif()
        endforeach()

        set(SD_DIR "${PATCHED_SD_ROOT}")
endif()

if (NOT EXISTS "${SD_DIR}/wan.hpp")
        message(FATAL_ERROR "stable-diffusion.cpp Wan headers not found at ${SD_DIR}. Run git submodule update --init --recursive (or disable overlays).")
endif()

option(WAN_SUPPORT "Enable Wan video generation support" ON)

# Configure stable-diffusion.cpp build options before adding the subdirectory
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "sd: build examples" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "sd: build shared libs" FORCE)
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "sd: build ggml as a separate shared lib" FORCE)
# Use the ggml submodule that ships with stable-diffusion.cpp to ensure API compatibility
set(SD_USE_SYSTEM_GGML OFF CACHE BOOL "sd: use system-installed GGML library" FORCE)
# Vulkan backend requires a host-side shader generator which must be
# built with a native host toolchain. That currently fails under the
# Android cross-build environment on Windows (host vs cross-compiler
# mismatch). Disable Vulkan by default to allow the project to build
# on developer machines. Enable Vulkan later when a proper host-tool
# solution is available.
set(SD_VULKAN ON CACHE BOOL "sd: vulkan backend" FORCE)
set(SD_CUDA OFF CACHE BOOL "sd: cuda backend" FORCE)
set(SD_METAL OFF CACHE BOOL "sd: metal backend" FORCE)
set(SD_OPENCL OFF CACHE BOOL "sd: opencl backend" FORCE)
set(SD_SYCL OFF CACHE BOOL "sd: sycl backend" FORCE)
set(SD_MUSA OFF CACHE BOOL "sd: musa backend" FORCE)

## Do not provide our own ggml target here; the stable-diffusion.cpp project
## will add its own ggml (submodule) with the required extended operators.

# Add stable-diffusion project and build its static lib target 'stable-diffusion'
# Ensure ggml-vulkan's ExternalProject (vulkan-shaders-gen) can find the correct build tool
if (CMAKE_MAKE_PROGRAM)
        # Write a minimal host toolchain that configures native host compiler
        set(_GGML_VK_HOST_TC "${CMAKE_CURRENT_BINARY_DIR}/ggml_vulkan_host_toolchain.cmake")
        file(WRITE "${_GGML_VK_HOST_TC}" "# Autogenerated by llmedge CMake\n"
                "set(CMAKE_SYSTEM_NAME \"${CMAKE_HOST_SYSTEM_NAME}\")\n"
                "set(CMAKE_SYSTEM_PROCESSOR \"${CMAKE_HOST_SYSTEM_PROCESSOR}\")\n"
                "set(CMAKE_BUILD_TYPE Release)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)\n"
                "set(CMAKE_MAKE_PROGRAM \"${CMAKE_MAKE_PROGRAM}\" CACHE FILEPATH \"\" FORCE)\n"
                "set(CMAKE_GENERATOR \"Ninja\" CACHE STRING \"\" FORCE)\n"
        )
        # Pass this toolchain to ggml so it uses our Ninja during host-side shader tool build
        set(GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "${_GGML_VK_HOST_TC}" CACHE STRING "Toolchain for ggml-vulkan shader generator" FORCE)
endif()
# Enable ggml Vulkan for stable-diffusion's ggml
set(GGML_VULKAN ON CACHE BOOL "ggml: vulkan" FORCE)
message(STATUS "Stable Diffusion Vulkan=${SD_VULKAN}; GGML Vulkan=${GGML_VULKAN}")

add_subdirectory(${SD_DIR} ${CMAKE_CURRENT_BINARY_DIR}/stable-diffusion.cpp)

# Add Vulkan-Hpp headers to ggml-vulkan target created by stable-diffusion.cpp
if (TARGET ggml-vulkan AND vulkan_hpp_SOURCE_DIR)
    target_include_directories(ggml-vulkan PRIVATE ${vulkan_hpp_SOURCE_DIR})
endif()

# JNI glue for stable diffusion
add_library(sdcpp SHARED
        sdcpp_jni.cpp
)

target_include_directories(sdcpp
        PUBLIC
        ${SD_DIR}
        ${SD_DIR}/thirdparty
)

target_compile_features(sdcpp PUBLIC c_std_11 cxx_std_17)

target_compile_options(sdcpp PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections -O3)

target_link_libraries(sdcpp
        android log
        stable-diffusion
)
if(SD_VULKAN)
        target_link_libraries(sdcpp vulkan)
endif()

if (WAN_SUPPORT)
        target_compile_definitions(sdcpp PRIVATE WAN_SUPPORT=1)
        message(STATUS "Wan video support is enabled for sdcpp")
endif()

target_link_options(sdcpp PRIVATE -Wl,--gc-sections -flto -Wl,--exclude-libs,ALL)

# ------------------------------------------------------------
# Whisper.cpp JNI wrapper build (Speech-to-Text)
# Build whisper with its OWN ggml sources compiled directly into whisper_jni
# to avoid target name conflicts and ensure API compatibility.
# Similar approach to bark_jni.
# ------------------------------------------------------------

# Resolve WHISPER_DIR to an absolute path
set(WHISPER_DIR_RELATIVE "../../../../whisper.cpp")
get_filename_component(WHISPER_DIR ${WHISPER_DIR_RELATIVE} ABSOLUTE)
set(WHISPER_GGML_DIR ${WHISPER_DIR}/ggml)

if (NOT EXISTS "${WHISPER_DIR}/include/whisper.h")
        message(FATAL_ERROR "whisper.cpp headers not found at ${WHISPER_DIR}. Please ensure whisper.cpp is cloned.")
endif()

# Whisper's bundled ggml sources (needed for CPU backend)
set(WHISPER_GGML_SOURCES
        ${WHISPER_GGML_DIR}/src/ggml.c
        ${WHISPER_GGML_DIR}/src/ggml-alloc.c
        ${WHISPER_GGML_DIR}/src/ggml-backend.cpp
        ${WHISPER_GGML_DIR}/src/ggml-backend-reg.cpp
        ${WHISPER_GGML_DIR}/src/ggml-opt.cpp
        ${WHISPER_GGML_DIR}/src/ggml-quants.c
        ${WHISPER_GGML_DIR}/src/ggml-threading.cpp
        ${WHISPER_GGML_DIR}/src/gguf.cpp
        # CPU backend
        ${WHISPER_GGML_DIR}/src/ggml-cpu/ggml-cpu.c
        ${WHISPER_GGML_DIR}/src/ggml-cpu/ggml-cpu.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/ops.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/vec.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/quants.c
        ${WHISPER_GGML_DIR}/src/ggml-cpu/traits.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/unary-ops.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/binary-ops.cpp
        ${WHISPER_GGML_DIR}/src/ggml-cpu/repack.cpp
        # Architecture-specific for ARM
        ${WHISPER_GGML_DIR}/src/ggml-cpu/arch/arm/quants.c
)

# Whisper source files
set(WHISPER_SOURCES
        ${WHISPER_DIR}/src/whisper.cpp
        whisper_jni.cpp
)

# All sources for whisper_jni
set(WHISPER_JNI_ALL_SOURCES
        ${WHISPER_GGML_SOURCES}
        ${WHISPER_SOURCES}
)

# Build whisper_jni library with all sources compiled in
add_library(whisper_jni SHARED ${WHISPER_JNI_ALL_SOURCES})

# CRITICAL: Add GGML_USE_CPU for proper CPU backend initialization on Android
# Define WHISPER_VERSION, GGML_VERSION and GGML_COMMIT which are required
# Enable OpenMP for multi-threaded ggml matrix operations
target_compile_definitions(whisper_jni PUBLIC GGML_USE_CPU)
target_compile_definitions(whisper_jni PRIVATE
        WHISPER_VERSION="1.0.0"
        GGML_VERSION="0.9.4"
        GGML_COMMIT="unknown"
        GGML_USE_OPENMP
)

# Set architecture-specific compile options for whisper
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        target_compile_options(whisper_jni PRIVATE -march=armv8.2-a+fp16)
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
        target_compile_options(whisper_jni PRIVATE -mfpu=neon-vfpv4)
endif()

# Include directories
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

# Link only against Android libraries (ggml is compiled in)
# Enable OpenMP for multi-threaded inference
target_link_libraries(whisper_jni
        android log
        -fopenmp -static-openmp
)

target_link_options(whisper_jni PRIVATE -Wl,--gc-sections -flto -Wl,--exclude-libs,ALL)

message(STATUS "Whisper.cpp JNI wrapper configured (direct source build with bundled ggml)")

message(STATUS "Whisper.cpp JNI wrapper configured (direct source build with bundled ggml, OpenMP enabled)")

# ------------------------------------------------------------
# Bark.cpp JNI wrapper build (Text-to-Speech)
# Build bark, encodec, and their ggml sources directly without add_subdirectory
# to avoid target name conflicts with stable-diffusion.cpp's ggml.
# ------------------------------------------------------------

# Bark numeric stability/perf tuning:
# Some aggressive compiler flags (especially reciprocal math / relaxed FP semantics or
# targeting newer ARM ISA extensions) can produce noticeably degraded audio (or SIGILL on
# older devices). Keep defaults conservative and allow opt-in.
option(LLMEDGE_BARK_AGGRESSIVE_ARM_OPT "Enable aggressive ARM tuning flags for Bark (may reduce correctness/compatibility)" OFF)

set(BARK_DIR_RELATIVE "../../../../bark.cpp")
get_filename_component(BARK_DIR ${BARK_DIR_RELATIVE} ABSOLUTE)

if (NOT EXISTS "${BARK_DIR}/bark.h")
        message(FATAL_ERROR "bark.cpp headers not found at ${BARK_DIR}. Please ensure bark.cpp is cloned.")
endif()

set(BARK_ENCODEC_DIR ${BARK_DIR}/encodec.cpp)
set(BARK_GGML_DIR ${BARK_ENCODEC_DIR}/ggml)

# Collect ggml sources from bark's bundled ggml (older API, compatible with bark/encodec)
set(BARK_GGML_SOURCES
        ${BARK_GGML_DIR}/src/ggml.c
        ${BARK_GGML_DIR}/src/ggml-alloc.c
        ${BARK_GGML_DIR}/src/ggml-backend.cpp
        ${BARK_GGML_DIR}/src/ggml-quants.c
        # ggml-aarch64.c is required for all ARM builds as it contains
        # implementations for quantize_mat_q8_0, ggml_gemv_*, ggml_gemm_*
        # that are referenced from ggml.c type_traits table
        ${BARK_GGML_DIR}/src/ggml-aarch64.c
)

# Encodec sources
set(ENCODEC_SOURCES
        ${BARK_ENCODEC_DIR}/encodec.cpp
        ${BARK_ENCODEC_DIR}/ops.cpp
)

# Bark sources
set(BARK_SOURCES
        ${BARK_DIR}/bark.cpp
)

# All sources for bark_jni
set(BARK_JNI_ALL_SOURCES
        ${BARK_GGML_SOURCES}
        ${ENCODEC_SOURCES}
        ${BARK_SOURCES}
        bark_jni.cpp
)

# Build bark_jni as a single shared library with all sources
add_library(bark_jni SHARED ${BARK_JNI_ALL_SOURCES})

target_include_directories(bark_jni
        PRIVATE
        ${BARK_DIR}
        ${BARK_ENCODEC_DIR}
        ${BARK_GGML_DIR}/include
        ${BARK_GGML_DIR}/src
)

# Define version constants to avoid linker errors
# GGML_USE_CPU is CRITICAL for proper CPU backend initialization on Android
target_compile_definitions(bark_jni
        PRIVATE
        GGML_COMMIT=""
        GGML_VERSION=""
        EXPORTING_BARK
        GGML_USE_CPU
)

target_compile_features(bark_jni PUBLIC c_std_11 cxx_std_17)

# Set architecture-specific compile options for bark
# Enable aggressive ARM optimizations for modern devices like S22 (Cortex-X2/A710/A510)
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        # Default to a conservative baseline to maximize device compatibility.
        target_compile_options(bark_jni PRIVATE -march=armv8-a)

        if (LLMEDGE_BARK_AGGRESSIVE_ARM_OPT)
                # Opt-in: faster on some devices, but may degrade numeric quality.
                # Avoid -ffast-math (incompatible with ggml), but even these flags can
                # change results; keep behind the switch.
                target_compile_options(bark_jni PRIVATE
                        -march=armv8.4-a+dotprod+fp16
                        -mtune=cortex-a78
                        -ffp-contract=fast
                        -fno-signed-zeros
                        -fno-trapping-math
                        -freciprocal-math
                )
        endif()
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
        target_compile_options(bark_jni PRIVATE -mfpu=neon-vfpv4)
endif()

target_compile_options(bark_jni PUBLIC
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -ffunction-sections
        -fdata-sections
        -O3
        -fopenmp
        -funroll-loops
)

# Enable OpenMP for multi-threaded inference - CRITICAL for performance
# Android NDK supports OpenMP via -static-openmp
target_compile_definitions(bark_jni PRIVATE GGML_USE_OPENMP)

target_link_libraries(bark_jni
        android log
        -fopenmp -static-openmp
)

target_link_options(bark_jni PRIVATE -Wl,--gc-sections -flto -Wl,--exclude-libs,ALL)

message(STATUS "Bark.cpp JNI wrapper configured (direct source build with OpenMP)")
