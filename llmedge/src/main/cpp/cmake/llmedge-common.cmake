# Shared llmedge native build configuration.
# The root CMakeLists.txt should set LLMEDGE_CPP_ROOT before including this file.

get_filename_component(LLAMA_DIR "${LLMEDGE_CPP_ROOT}/../../../../llama.cpp" ABSOLUTE)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

option(LLMEDGE_ANDROID_OPENCL "Enable experimental Android OpenCL backend support" OFF)

set(GGML_DIR ${LLAMA_DIR}/ggml)
set(COMMON_DIR ${LLAMA_DIR}/common)
set(VENDOR_DIR ${LLAMA_DIR}/vendor)
set(SMOLLM_SOURCES
        ${GGML_DIR}/src/ggml-alloc.c
        ${GGML_DIR}/src/ggml-backend.cpp
        ${GGML_DIR}/src/ggml-threading.cpp
        ${GGML_DIR}/src/ggml-quants.c
        ${GGML_DIR}/src/ggml-backend-reg.cpp
        ${GGML_DIR}/src/ggml-backend-dl.cpp
        ${GGML_DIR}/src/ggml-opt.cpp
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

        # llama.cpp core sources are globbed below (LLAMA_CPP_SOURCES) to tolerate upstream file renames.

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

        # llmedge-local build info for llama.cpp common
        ${LLMEDGE_CPP_ROOT}/llama_build_info.cpp

        ${LLMEDGE_CPP_ROOT}/LLMInference.cpp
        ${LLMEDGE_CPP_ROOT}/smollm.cpp
        # libmtmd (multimodal projector) from llama.cpp
        ${LLAMA_DIR}/tools/mtmd/mtmd.cpp
        ${LLAMA_DIR}/tools/mtmd/mtmd-helper.cpp
        ${LLAMA_DIR}/tools/mtmd/mtmd-audio.cpp
        ${LLAMA_DIR}/tools/mtmd/clip.cpp
)

if (${ANDROID_ABI} STREQUAL "arm64-v8a" OR ${ANDROID_ABI} STREQUAL "armeabi-v7a")
        list(APPEND SMOLLM_SOURCES
                ${GGML_DIR}/src/ggml-cpu/arch/arm/quants.c
                ${GGML_DIR}/src/ggml-cpu/arch/arm/repack.cpp
        )
elseif (${ANDROID_ABI} STREQUAL "x86" OR ${ANDROID_ABI} STREQUAL "x86_64")
        list(APPEND SMOLLM_SOURCES
                ${GGML_DIR}/src/ggml-cpu/arch/x86/quants.c
                ${GGML_DIR}/src/ggml-cpu/arch/x86/repack.cpp
        )
endif()

# mtmd has additional model-specific graph implementations under tools/mtmd/models/*.cpp
file(GLOB MTMD_MODEL_SOURCES "${LLAMA_DIR}/tools/mtmd/models/*.cpp")
if(MTMD_MODEL_SOURCES)
        list(APPEND SMOLLM_SOURCES ${MTMD_MODEL_SOURCES})
endif()

file(GLOB LLAMA_CPP_SOURCES
        "${LLAMA_DIR}/src/*.cpp"
        "${LLAMA_DIR}/src/*.c"
)
if(LLAMA_CPP_SOURCES)
        list(APPEND SMOLLM_SOURCES ${LLAMA_CPP_SOURCES})
else()
        message(FATAL_ERROR "No llama.cpp sources found under ${LLAMA_DIR}/src")
endif()

# If llama.cpp contains per-model source files, append them to SMOLLM_SOURCES.
# Newer llama.cpp versions keep many model-specific builders in src/models/*.cpp; without
# these, link errors like llm_build_* will occur.
if(EXISTS "${LLAMA_DIR}/src/models")
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

# ------------------------------------------------------------
# OpenCL configuration (Android / arm64 only)
# ------------------------------------------------------------
set(LLMEDGE_OPENCL_ENABLED OFF)
set(LLMEDGE_OPENCL_HEADERS_DIR "")

if (LLMEDGE_ANDROID_OPENCL)
    if (NOT ANDROID)
        message(STATUS "LLMEDGE_ANDROID_OPENCL requested on a non-Android build; disabling")
    elseif (NOT ANDROID_ABI STREQUAL "arm64-v8a")
        message(STATUS "LLMEDGE_ANDROID_OPENCL is only supported for Android arm64-v8a; disabling for ${ANDROID_ABI}")
    else()
        include(FetchContent)

        FetchContent_Declare(
            opencl_headers
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
            GIT_TAG        v2024.10.24
            GIT_SHALLOW    TRUE
        )
        FetchContent_GetProperties(opencl_headers)
        if (NOT opencl_headers_POPULATED)
            FetchContent_Populate(opencl_headers)
        endif()

        set(LLMEDGE_OPENCL_HEADERS_DIR "${opencl_headers_SOURCE_DIR}")
        set(OPENCL_ICD_LOADER_HEADERS_DIR "${LLMEDGE_OPENCL_HEADERS_DIR}" CACHE PATH "OpenCL headers for Android loader build" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(
            opencl_icd_loader
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
            GIT_TAG        v2024.10.24
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(opencl_icd_loader)

        if (TARGET OpenCL AND NOT TARGET OpenCL::OpenCL)
            add_library(OpenCL::OpenCL ALIAS OpenCL)
        endif()

        if (TARGET OpenCL::OpenCL)
            set(LLMEDGE_OPENCL_ENABLED ON)
            message(STATUS "Experimental Android OpenCL backend support enabled")
        else()
            message(WARNING "LLMEDGE_ANDROID_OPENCL was requested, but OpenCL::OpenCL was not created by OpenCL-ICD-Loader")
        endif()
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

function(llmedge_enable_android_opencl target_name ggml_root)
    if (NOT LLMEDGE_OPENCL_ENABLED)
        return()
    endif()

    find_package(Python3 REQUIRED)

    set(_ggml_opencl_dir "${ggml_root}/src/ggml-opencl")
    set(_kernel_dir "${_ggml_opencl_dir}/kernels")
    set(_embed_script "${_kernel_dir}/embed_kernel.py")
    set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/${target_name}-ggml-opencl")
    file(MAKE_DIRECTORY "${_generated_dir}")

    file(GLOB _kernel_sources "${_kernel_dir}/*.cl")
    set(_generated_headers)
    foreach(_kernel IN LISTS _kernel_sources)
        get_filename_component(_kernel_name "${_kernel}" NAME_WE)
        set(_generated_header "${_generated_dir}/${_kernel_name}.cl.h")
        add_custom_command(
            OUTPUT "${_generated_header}"
            COMMAND ${Python3_EXECUTABLE} "${_embed_script}" "${_kernel}" "${_generated_header}"
            DEPENDS "${_kernel}" "${_embed_script}"
            COMMENT "Embedding OpenCL kernel ${_kernel_name} for ${target_name}"
            VERBATIM
        )
        list(APPEND _generated_headers "${_generated_header}")
    endforeach()

    target_sources(
        ${target_name}
        PRIVATE
        "${_ggml_opencl_dir}/ggml-opencl.cpp"
        ${_generated_headers}
    )
    target_include_directories(
        ${target_name}
        PRIVATE
        "${ggml_root}/include"
        "${ggml_root}/src"
        "${_generated_dir}"
        "${LLMEDGE_OPENCL_HEADERS_DIR}"
    )
    target_compile_definitions(
        ${target_name}
        PRIVATE
        GGML_USE_OPENCL
        GGML_OPENCL_SOA_Q
        GGML_OPENCL_EMBED_KERNELS
        GGML_OPENCL_USE_ADRENO_KERNELS
        GGML_OPENCL_TARGET_VERSION=300
    )
    target_link_libraries(${target_name} OpenCL::OpenCL)
endfunction()

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
            ${LLAMA_DIR}/src
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
            android log dl
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
    set(_ggml_cpu_flags -DGGML_USE_CPU)
    if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        list(APPEND _ggml_cpu_flags -DGGML_USE_CPU_AARCH64)
    endif()
    target_compile_options(
            ${target_name}
            PUBLIC
            ${_ggml_cpu_flags} -O3 -funroll-loops
    )
endfunction()
