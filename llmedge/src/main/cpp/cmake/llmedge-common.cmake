# Shared llmedge native build configuration.
# The root CMakeLists.txt should set LLMEDGE_CPP_ROOT before including this file.

get_filename_component(LLAMA_DIR "${LLMEDGE_CPP_ROOT}/../../../../llama.cpp" ABSOLUTE)

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

        ${LLMEDGE_CPP_ROOT}/LLMInference.cpp
        ${LLMEDGE_CPP_ROOT}/smollm.cpp
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
        ${LLMEDGE_CPP_ROOT}/GGUFReader.cpp
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
    # As we are NOT using llama.cpp's CMake script, we need to define them here.
    target_compile_definitions(
            ${target_name}
            PRIVATE
            GGML_COMMIT=""
            GGML_VERSION=""
    )

    target_compile_options(
            ${target_name}
            PUBLIC
            -fvisibility=hidden -fvisibility-inlines-hidden
    )
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
