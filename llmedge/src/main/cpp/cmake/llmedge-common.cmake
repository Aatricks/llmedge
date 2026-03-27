# Shared llmedge native build configuration.
# The root CMakeLists.txt should set LLMEDGE_CPP_ROOT before including this file.

include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-mods.cmake")

get_filename_component(LLAMA_DIR_UPSTREAM "${LLMEDGE_CPP_ROOT}/../../../../llama.cpp" ABSOLUTE)
get_filename_component(LLMEDGE_REPO_ROOT "${LLMEDGE_CPP_ROOT}/../../../.." ABSOLUTE)
set(LLMEDGE_MODS_DIR "${LLMEDGE_REPO_ROOT}/mods")
llmedge_prepare_patch_modded_tree(
        NAME "llama.cpp"
        SOURCE_ROOT "${LLAMA_DIR_UPSTREAM}"
        MODS_ROOT "${LLMEDGE_MODS_DIR}/llama.cpp"
        USE_MODS_VAR "LLMEDGE_LLAMA_USE_MODS"
        OUT_VAR LLAMA_DIR
)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

option(LLMEDGE_ANDROID_OPENCL "Enable experimental Android OpenCL backend support" OFF)

set(GGML_DIR ${LLAMA_DIR}/ggml)
set(COMMON_DIR ${LLAMA_DIR}/common)
set(VENDOR_DIR ${LLAMA_DIR}/vendor)
if(EXISTS "${LLAMA_DIR}/tools/mtmd/mtmd.cpp")
        set(MTMD_DIR "${LLAMA_DIR}/tools/mtmd")
elseif(EXISTS "${LLAMA_DIR}/examples/mtmd/mtmd.cpp")
        set(MTMD_DIR "${LLAMA_DIR}/examples/mtmd")
else()
        message(FATAL_ERROR "No compatible mtmd sources found under ${LLAMA_DIR}")
endif()

set(LLMEDGE_LLAMA_USE_MEMORY_API OFF CACHE BOOL "Use llama_memory_* APIs instead of llama_kv_cache_* compatibility APIs" FORCE)
file(GLOB LLMEDGE_COMMON_SOURCES CONFIGURE_DEPENDS
        "${COMMON_DIR}/*.cpp"
        "${COMMON_DIR}/jinja/*.cpp"
)
set(LLMEDGE_IQK_BASE_SOURCES
        ${GGML_DIR}/src/iqk/iqk_quantize.cpp
        ${GGML_DIR}/src/iqk/iqk_cpu_ops.cpp
        ${GGML_DIR}/src/iqk/iqk_mul_mat.cpp
        ${GGML_DIR}/src/iqk/iqk_flash_attn.cpp
)
set(LLMEDGE_IQK_DOTPROD_SOURCES
        ${GGML_DIR}/src/iqk/iqk_gemm_floats.cpp
        ${GGML_DIR}/src/iqk/iqk_gemm_kquants.cpp
        ${GGML_DIR}/src/iqk/iqk_gemm_ktquants.cpp
        ${GGML_DIR}/src/iqk/iqk_gemm_iquants.cpp
        ${GGML_DIR}/src/iqk/iqk_gemm_iqk_quants.cpp
        ${GGML_DIR}/src/iqk/iqk_gemm_1bit.cpp
        ${GGML_DIR}/src/iqk/iqk_gemm_legacy_quants.cpp
        ${GGML_DIR}/src/iqk/fa/iqk_fa_576_512.cpp
        ${GGML_DIR}/src/iqk/fa/iqk_fa_320_256.cpp
        ${GGML_DIR}/src/iqk/fa/iqk_fa_192_128.cpp
        ${GGML_DIR}/src/iqk/fa/iqk_fa_192_192.cpp
        ${GGML_DIR}/src/iqk/fa/iqk_fa_256_256.cpp
        ${GGML_DIR}/src/iqk/fa/iqk_fa_128_128.cpp
        ${GGML_DIR}/src/iqk/fa/iqk_fa_96_96.cpp
        ${GGML_DIR}/src/iqk/fa/iqk_fa_64_64.cpp
)
# The patched IQK sources rely on IQK_FORCE_IMPLEMENT in two cases:
# - desktop x86 host builds, where upstream gating does not enable the implementation by default
# - Android arm64 targets such as ggufreader and smollm_v8, which compile IQK sources without
#   dotprod-specific march flags but still need the generic AArch64 NEON helper implementations
if ((NOT ANDROID AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        OR (ANDROID AND "${ANDROID_ABI}" STREQUAL "arm64-v8a"))
        set_source_files_properties(
                ${GGML_DIR}/src/iqk/iqk_cpu_ops.cpp
                PROPERTIES COMPILE_DEFINITIONS "IQK_FORCE_IMPLEMENT"
        )
endif()
set(LLMEDGE_GGML_CORE_SOURCES
        ${GGML_DIR}/src/ggml-alloc.c
        ${GGML_DIR}/src/ggml-backend.cpp
        ${GGML_DIR}/src/ggml-quants.c
        ${GGML_DIR}/src/ggml.c
)
set(LLMEDGE_GGML_VULKAN_SOURCES)
set(LLMEDGE_GGML_EXTRA_INCLUDES)
set(LLMEDGE_GGML_EXTRA_LIBS)
set(LLMEDGE_GGML_EXTRA_DEFINITIONS)
set(LLMEDGE_GGML_ARCH_SOURCES)
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        list(APPEND LLMEDGE_GGML_ARCH_SOURCES ${GGML_DIR}/src/ggml-aarch64.c)
endif()
set(SMOLLM_SOURCES
        ${LLMEDGE_GGML_CORE_SOURCES}
        ${LLMEDGE_GGML_ARCH_SOURCES}

        # llama.cpp core sources are globbed below (LLAMA_CPP_SOURCES) to tolerate upstream file renames.

        ${LLMEDGE_COMMON_SOURCES}

        ${LLMEDGE_CPP_ROOT}/LLMInference.cpp
        ${LLMEDGE_CPP_ROOT}/smollm.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_completion.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_embeddings.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_load.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_state.cpp
        # libmtmd (multimodal projector) from llama.cpp
        ${MTMD_DIR}/mtmd.cpp
        ${MTMD_DIR}/mtmd-helper.cpp
        ${MTMD_DIR}/mtmd-audio.cpp
        ${MTMD_DIR}/clip.cpp
)

# mtmd may provide additional model-specific graph implementations under models/*.cpp
file(GLOB MTMD_MODEL_SOURCES "${MTMD_DIR}/models/*.cpp")
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

    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "GGML_VULKAN is enabled but glslc was not found")
    endif()

    function(llmedge_test_shader_extension_support extension_name test_shader_file result_variable)
        execute_process(
                COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${test_shader_file}"
                OUTPUT_QUIET
                ERROR_VARIABLE glslc_error
        )

        if("${glslc_error}" MATCHES ".*extension not supported: ${extension_name}.*")
            set(${result_variable} OFF PARENT_SCOPE)
        else()
            set(${result_variable} ON PARENT_SCOPE)
        endif()
    endfunction()

    set(_llmedge_vk_shader_gen_cmake_args)
    if (NOT GGML_VULKAN_NO_COOPMAT)
        llmedge_test_shader_extension_support(
                "GL_KHR_cooperative_matrix"
                "${GGML_DIR}/src/vulkan-shaders/test_coopmat_support.comp"
                "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"
        )
        if (GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT=ON)
        endif()
    endif()

    if (NOT GGML_VULKAN_NO_COOPMAT2)
        llmedge_test_shader_extension_support(
                "GL_NV_cooperative_matrix2"
                "${GGML_DIR}/src/vulkan-shaders/test_coopmat2_support.comp"
                "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"
        )
        if (GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_COOPMAT2_GLSLC_SUPPORT=ON)
        endif()
    endif()

    if (NOT GGML_VULKAN_NO_INT_DOT)
        llmedge_test_shader_extension_support(
                "GL_EXT_integer_dot_product"
                "${GGML_DIR}/src/vulkan-shaders/test_integer_dot_support.comp"
                "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"
        )
        if (GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT=ON)
        endif()
    endif()

    if (NOT GGML_VULKAN_NO_BF16)
        llmedge_test_shader_extension_support(
                "GL_EXT_bfloat16"
                "${GGML_DIR}/src/vulkan-shaders/test_bfloat16_support.comp"
                "GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"
        )
        if (GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_BFLOAT16_GLSLC_SUPPORT=ON)
        endif()
    endif()

    list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_USE_VULKAN)
    if (GGML_VULKAN_CHECK_RESULTS)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_CHECK_RESULTS)
    endif()
    if (GGML_VULKAN_DEBUG)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_DEBUG)
    endif()
    if (GGML_VULKAN_MEMORY_DEBUG)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_MEMORY_DEBUG)
    endif()
    if (GGML_VULKAN_SHADER_DEBUG_INFO)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_SHADER_DEBUG_INFO)
        list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_SHADER_DEBUG_INFO=ON)
    endif()
    if (GGML_VULKAN_PERF)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_PERF)
    endif()
    if (GGML_VULKAN_VALIDATE)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_VALIDATE)
    endif()
    if (GGML_VULKAN_RUN_TESTS)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_RUN_TESTS)
    endif()

    if (CMAKE_MAKE_PROGRAM)
        set(_LLMEDGE_GGML_VK_HOST_TC "${CMAKE_CURRENT_BINARY_DIR}/ggml_vulkan_host_toolchain.cmake")
        file(WRITE "${_LLMEDGE_GGML_VK_HOST_TC}" "# Autogenerated by llmedge CMake\n"
                "set(CMAKE_SYSTEM_NAME \"${CMAKE_HOST_SYSTEM_NAME}\")\n"
                "set(CMAKE_SYSTEM_PROCESSOR \"${CMAKE_HOST_SYSTEM_PROCESSOR}\")\n"
                "set(CMAKE_BUILD_TYPE Release)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)\n"
                "set(CMAKE_MAKE_PROGRAM \"${CMAKE_MAKE_PROGRAM}\" CACHE FILEPATH \"\" FORCE)\n"
                "set(CMAKE_GENERATOR \"Ninja\" CACHE STRING \"\" FORCE)\n"
        )
        set(GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "${_LLMEDGE_GGML_VK_HOST_TC}" CACHE STRING "Toolchain for ggml-vulkan shader generator" FORCE)
    endif()

    include(ExternalProject)
    set(_llmedge_vk_shader_gen_install_dir "${CMAKE_CURRENT_BINARY_DIR}/llmedge-vulkan-shaders-gen")
    set(_llmedge_vk_shader_gen_build_dir "${CMAKE_CURRENT_BINARY_DIR}/llmedge-vulkan-shaders-gen-build")
    set(_llmedge_vk_shader_gen_args
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_INSTALL_BINDIR=.
            -DCMAKE_BUILD_TYPE=Release
    )
    if (CMAKE_CROSSCOMPILING AND GGML_VULKAN_SHADERS_GEN_TOOLCHAIN)
        list(APPEND _llmedge_vk_shader_gen_args -DCMAKE_TOOLCHAIN_FILE=${GGML_VULKAN_SHADERS_GEN_TOOLCHAIN})
    endif()
    list(APPEND _llmedge_vk_shader_gen_args ${_llmedge_vk_shader_gen_cmake_args})

    ExternalProject_Add(
            llmedge-text-vulkan-shaders-gen
            SOURCE_DIR "${GGML_DIR}/src/vulkan-shaders"
            BINARY_DIR "${_llmedge_vk_shader_gen_build_dir}"
            INSTALL_DIR "${_llmedge_vk_shader_gen_install_dir}"
            CMAKE_ARGS ${_llmedge_vk_shader_gen_args}
            BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
            INSTALL_COMMAND ${CMAKE_COMMAND} -E env --unset=DESTDIR ${CMAKE_COMMAND} --install . --config Release
    )

    set(_llmedge_vk_host_suffix "")
    if (CMAKE_HOST_WIN32)
        set(_llmedge_vk_host_suffix ".exe")
    endif()
    set(_llmedge_vk_genshaders_cmd "${_llmedge_vk_shader_gen_install_dir}/vulkan-shaders-gen${_llmedge_vk_host_suffix}")
    set(_llmedge_vk_header "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp")
    set(_llmedge_vk_source "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.cpp")
    set(_llmedge_vk_input_dir "${GGML_DIR}/src/vulkan-shaders")
    set(_llmedge_vk_output_dir "${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv")
    file(GLOB _llmedge_vk_shader_files CONFIGURE_DEPENDS "${_llmedge_vk_input_dir}/*.comp")

    add_custom_command(
            OUTPUT "${_llmedge_vk_header}" "${_llmedge_vk_source}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_llmedge_vk_output_dir}"
            COMMAND "${_llmedge_vk_genshaders_cmd}"
                    --glslc "${Vulkan_GLSLC_EXECUTABLE}"
                    --input-dir "${_llmedge_vk_input_dir}"
                    --output-dir "${_llmedge_vk_output_dir}"
                    --target-hpp "${_llmedge_vk_header}"
                    --target-cpp "${_llmedge_vk_source}"
                    --no-clean
            DEPENDS ${_llmedge_vk_shader_files} llmedge-text-vulkan-shaders-gen
            COMMENT "Generate Vulkan shaders for llmedge text runtime"
    )

    list(APPEND LLMEDGE_GGML_VULKAN_SOURCES
            ${GGML_DIR}/src/ggml-vulkan.cpp
            "${_llmedge_vk_source}"
    )
    list(APPEND LLMEDGE_GGML_EXTRA_INCLUDES
            ${Vulkan_INCLUDE_DIRS}
            "${CMAKE_CURRENT_BINARY_DIR}"
    )
    list(APPEND LLMEDGE_GGML_EXTRA_LIBS "${VULKAN_LIB}")
endif()

if(LLMEDGE_GGML_VULKAN_SOURCES)
        list(APPEND SMOLLM_SOURCES ${LLMEDGE_GGML_VULKAN_SOURCES})
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
        ${LLMEDGE_GGML_CORE_SOURCES}
        ${LLMEDGE_GGML_ARCH_SOURCES}
        ${LLMEDGE_CPP_ROOT}/gguf_reader_internal.cpp
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
    target_compile_features(${target_name} PUBLIC c_std_11 cxx_std_20)
    target_include_directories(
            ${target_name}
            PUBLIC
            ${COMMON_DIR}
            ${GGML_DIR}/include
            ${GGML_DIR}/src
            ${GGML_DIR}/src/iqk
            ${LLMEDGE_GGML_EXTRA_INCLUDES}
            ${LLAMA_DIR}/include
            ${LLAMA_DIR}/src
            ${MTMD_DIR}
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
            ${LLMEDGE_GGML_EXTRA_DEFINITIONS}
            LLMEDGE_LLAMA_USE_MEMORY_API=$<BOOL:${LLMEDGE_LLAMA_USE_MEMORY_API}>
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
            ${LLMEDGE_GGML_EXTRA_LIBS}
    )

    target_link_options(
            ${target_name}
            PRIVATE
            -Wl,--gc-sections -flto
            -Wl,--exclude-libs,ALL
    )
endfunction()

function(enable_iqk_target target_name)
    target_sources(${target_name} PRIVATE ${LLMEDGE_IQK_BASE_SOURCES})
    target_compile_definitions(
            ${target_name}
            PRIVATE
            GGML_USE_IQK_MULMAT
            GGML_IQK_FLASH_ATTENTION
    )
endfunction()

function(build_library_arm64 target_name cpu_flags)
    build_library(${target_name})
    enable_iqk_target(${target_name})
    if("${cpu_flags}" MATCHES "dotprod")
        target_sources(${target_name} PRIVATE ${LLMEDGE_IQK_DOTPROD_SOURCES})
    endif()
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
        enable_iqk_target(${target_name})
        list(APPEND _ggml_cpu_flags -DGGML_USE_CPU_AARCH64)
    endif()
    target_compile_options(
            ${target_name}
            PUBLIC
            ${_ggml_cpu_flags} -O3 -funroll-loops
    )
endfunction()
